"""Positive-first curriculum hooks for Miles agentic GRPO training.

The rollout sample filter preserves the existing token/context truncation mask
and, during a zero-negative-weight phase, masks samples whose group-centered
advantage is non-positive.  The reward postprocessor computes group-centered
GRPO rewards by ``Sample.group_index`` and scales non-positive advantages by a
rollout-dependent coefficient.

Configure the rollout schedule through ``POSITIVE_FIRST_NEGATIVE_SCHEDULE``::

    0:0,43:0.25,47:0.5,56:1.0

Each entry is ``start_rollout:negative_weight``.  Starts must be strictly
increasing, the first start must be zero, and weights must lie in [0, 1].
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from collections.abc import Sequence

import torch

from miles.utils.types import Sample

logger = logging.getLogger(__name__)

SCHEDULE_ENV = "POSITIVE_FIRST_NEGATIVE_SCHEDULE"
TRUNCATION_EXIT_STATUSES = frozenset(
    {
        "BadRequestError",
        "ContextWindowExceededError",
        "OutputLengthExceededError",
    }
)


def _parse_schedule(raw: str) -> tuple[tuple[int, float], ...]:
    if not raw or not raw.strip():
        raise ValueError(f"{SCHEDULE_ENV} must be set to a non-empty start:weight schedule")

    entries: list[tuple[int, float]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item or ":" not in item:
            raise ValueError(
                f"invalid {SCHEDULE_ENV} entry {item!r}; expected comma-separated start:weight pairs"
            )
        start_text, weight_text = item.split(":", 1)
        try:
            start = int(start_text)
            weight = float(weight_text)
        except ValueError as exc:
            raise ValueError(f"invalid {SCHEDULE_ENV} entry {item!r}") from exc
        if start < 0:
            raise ValueError(f"{SCHEDULE_ENV} rollout starts must be non-negative, got {start}")
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError(f"{SCHEDULE_ENV} weights must be finite and in [0, 1], got {weight}")
        if entries and start <= entries[-1][0]:
            raise ValueError(f"{SCHEDULE_ENV} rollout starts must be strictly increasing")
        entries.append((start, weight))

    if entries[0][0] != 0:
        raise ValueError(f"{SCHEDULE_ENV} must begin at rollout 0")
    return tuple(entries)


def _configured_schedule() -> tuple[tuple[int, float], ...]:
    return _parse_schedule(os.environ.get(SCHEDULE_ENV, ""))


def _negative_weight(rollout_id: int, schedule: Sequence[tuple[int, float]]) -> float:
    weight = schedule[0][1]
    for start, candidate in schedule:
        if rollout_id < start:
            break
        weight = candidate
    return weight


def _flat_samples(samples) -> list[Sample]:
    flat: list[Sample] = []
    for sample in samples:
        if isinstance(sample, list):
            flat.extend(_flat_samples(sample))
        else:
            flat.append(sample)
    if not flat:
        raise ValueError("positive-first curriculum received an empty sample batch")
    if not all(isinstance(sample, Sample) for sample in flat):
        raise TypeError("positive-first curriculum expects Miles Sample objects")
    return flat


def _batch_rollout_id(samples: Sequence[Sample]) -> int:
    rollout_ids = {(sample.metadata or {}).get("rollout_id") for sample in samples}
    if None in rollout_ids:
        raise ValueError("positive-first curriculum requires metadata['rollout_id'] on every sample")
    if len(rollout_ids) != 1:
        rollout_id_list = ", ".join(sorted(repr(value) for value in rollout_ids))
        raise ValueError(f"positive-first curriculum batch spans multiple rollout IDs: {rollout_id_list}")
    rollout_id = next(iter(rollout_ids))
    if isinstance(rollout_id, bool) or not isinstance(rollout_id, int) or rollout_id < 0:
        raise ValueError(f"invalid rollout_id for positive-first curriculum: {rollout_id!r}")
    return rollout_id


def _raw_rewards(args, samples: Sequence[Sample]) -> list[float]:
    rewards: list[float] = []
    for sample in samples:
        value = sample.get_reward_value(args)
        try:
            reward = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric reward in positive-first curriculum: {value!r}") from exc
        if not math.isfinite(reward):
            raise ValueError(f"non-finite reward in positive-first curriculum: {reward!r}")
        rewards.append(reward)
    return rewards


def _group_indices(args, samples: Sequence[Sample]) -> dict[int, list[int]]:
    expected_group_size = getattr(args, "n_samples_per_prompt", None)
    if (
        isinstance(expected_group_size, bool)
        or not isinstance(expected_group_size, int)
        or expected_group_size <= 0
    ):
        raise ValueError(
            "positive-first curriculum requires a positive integer n_samples_per_prompt; "
            f"got {expected_group_size!r}"
        )

    groups: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        group_index = sample.group_index
        if isinstance(group_index, bool) or not isinstance(group_index, int):
            raise ValueError(
                "positive-first curriculum requires an integer group_index on every sample; "
                f"got {group_index!r}"
            )
        groups[group_index].append(index)

    incomplete_groups = {
        group_index: len(indices)
        for group_index, indices in groups.items()
        if len(indices) != expected_group_size
    }
    if incomplete_groups:
        details = ", ".join(
            f"{group_index}:{size}" for group_index, size in sorted(incomplete_groups.items())
        )
        raise ValueError(
            "positive-first curriculum requires complete prompt groups of "
            f"n_samples_per_prompt={expected_group_size}; got {details}"
        )
    return groups


def _processed_rewards(
    args, samples: Sequence[Sample]
) -> tuple[list[float], list[float], list[float], int, float]:
    if getattr(args, "advantage_estimator", None) not in {"grpo", "gspo"}:
        raise ValueError("positive-first curriculum requires the GRPO or GSPO advantage estimator")
    if not getattr(args, "rewards_normalization", True):
        raise ValueError("positive-first curriculum requires group reward centering")
    if getattr(args, "normalize_advantages", False):
        raise ValueError(
            "positive-first curriculum is incompatible with downstream advantage normalization"
        )

    raw_rewards = _raw_rewards(args, samples)
    rollout_id = _batch_rollout_id(samples)
    negative_weight = _negative_weight(rollout_id, _configured_schedule())

    centered = [0.0] * len(samples)
    for indices in _group_indices(args, samples).values():
        group_rewards = torch.tensor([raw_rewards[index] for index in indices], dtype=torch.float32)
        group_values = group_rewards - group_rewards.mean()
        if getattr(args, "grpo_std_normalization", False):
            if len(indices) < 2:
                raise ValueError("GRPO std normalization requires at least two samples per group")
            group_values = group_values / (group_values.std() + 1e-6)
        for index, value in zip(indices, group_values.tolist(), strict=True):
            centered[index] = value

    processed = [value if value > 0.0 else value * negative_weight for value in centered]
    return raw_rewards, centered, processed, rollout_id, negative_weight


def positive_first_sample_filter(args, samples: list[Sample]) -> None:
    """Compose token truncation masking with the positive-only phase mask."""
    flat = _flat_samples(samples)
    truncation_masked_count = 0
    for sample in flat:
        exit_status = (sample.metadata or {}).get("exit_status", "")
        if exit_status in TRUNCATION_EXIT_STATUSES or sample.status == Sample.Status.TRUNCATED:
            sample.remove_sample = True
            truncation_masked_count += 1

    _, centered, processed, rollout_id, negative_weight = _processed_rewards(args, flat)

    positive_count = 0
    negative_count = 0
    zero_count = 0
    curriculum_masked_count = 0
    for sample, centered_value, processed_value in zip(flat, centered, processed, strict=True):
        sample.metadata["curriculum_negative_weight"] = negative_weight
        sample.metadata["curriculum_centered_reward"] = centered_value
        sample.metadata["curriculum_processed_reward"] = processed_value
        sample.metadata["curriculum_truncation_masked"] = bool(
            (sample.metadata or {}).get("exit_status", "") in TRUNCATION_EXIT_STATUSES
            or sample.status == Sample.Status.TRUNCATED
        )
        sample.metadata["curriculum_masked"] = False
        if centered_value > 0.0:
            positive_count += 1
        else:
            if centered_value < 0.0:
                negative_count += 1
            else:
                zero_count += 1
            if negative_weight == 0.0:
                if not sample.remove_sample:
                    curriculum_masked_count += 1
                    sample.metadata["curriculum_masked"] = True
                sample.remove_sample = True

    logger.info(
        "Positive-first sample filter: rollout=%d negative_weight=%.3f positives=%d "
        "negatives=%d zeros=%d curriculum_masked=%d truncation_masked=%d active=%d/%d",
        rollout_id,
        negative_weight,
        positive_count,
        negative_count,
        zero_count,
        curriculum_masked_count,
        truncation_masked_count,
        sum(not sample.remove_sample for sample in flat),
        len(flat),
    )


def positive_first_reward_postprocess(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """Return raw rewards and group-centered, curriculum-weighted rewards."""
    flat = _flat_samples(samples)
    if len(flat) != len(samples):
        raise ValueError("reward postprocessor expects a flat sample list")

    raw_rewards, centered, processed, rollout_id, negative_weight = _processed_rewards(args, flat)
    positive_count = 0
    negative_count = 0
    zero_count = 0
    for sample, centered_value, value in zip(flat, centered, processed, strict=True):
        sample.metadata["curriculum_negative_weight"] = negative_weight
        sample.metadata["curriculum_centered_reward"] = centered_value
        sample.metadata["curriculum_processed_reward"] = value
        if centered_value > 0.0:
            positive_count += 1
        else:
            if centered_value < 0.0:
                negative_count += 1
            else:
                zero_count += 1
            if negative_weight == 0.0:
                sample.remove_sample = True

    negative_pre_scale = [abs(value) for value in centered if value < 0.0]
    negative_post_scale = [abs(value) for value in processed if value < 0.0]
    logger.info(
        "Positive-first reward postprocess: rollout=%d negative_weight=%.3f groups=%d "
        "positives=%d negatives=%d zeros=%d negative_abs_mean_pre=%.6f "
        "negative_abs_mean_post=%.6f active=%d/%d",
        rollout_id,
        negative_weight,
        len({sample.group_index for sample in flat}),
        positive_count,
        negative_count,
        zero_count,
        sum(negative_pre_scale) / len(negative_pre_scale) if negative_pre_scale else 0.0,
        sum(negative_post_scale) / len(negative_post_scale) if negative_post_scale else 0.0,
        sum(not sample.remove_sample for sample in flat),
        len(flat),
    )
    return raw_rewards, processed
