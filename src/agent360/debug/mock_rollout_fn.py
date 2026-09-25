# Modified for release: configurable deployment paths and public source references.
"""Mock Miles rollout function for fast train_step debugging.

Generate synthetic samples for regression-testing these failure modes:

  1. None-reward ABORTED samples (exercises radixark/miles#989 fix to
     `_compute_zero_std_metrics`; without the fix, round(None) crashes the
     RolloutManager actor after rollouts have been collected, silently
     stalling train_step).
  2. Full-None groups (confirms the 'none' bucket label path).

Wire via:
  --rollout-function-path agent360.debug.mock_rollout_fn.generate_rollout
  --custom-rm-path        agent360.debug.mock_rollout_fn.reward_func

Environment knobs (read at call time, not import):
  MOCK_ROLLOUT_ABORT_FRAC        float in [0,1]   default 0.5
  MOCK_ROLLOUT_NONE_REWARD       "1" to use None  default "1"  (force the
                                                                round(None)
                                                                code path;
                                                                set "0" for
                                                                0.0)
  MOCK_ROLLOUT_FORCE_NONE_GROUP  "1" to make at least one group fully-None
  MOCK_ROLLOUT_SLEEP_SECS        float             simulated rollout wall-time
                                                    (default 0.5s)
"""

from __future__ import annotations

import os
import random
import time
from argparse import Namespace
from typing import Any

from miles.rollout.base_types import RolloutFnTrainOutput, RolloutFnEvalOutput
from miles.rollout.sglang_rollout import GenerateState
from miles.utils.types import Sample


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0") == "1"


def _build_completed_sample(sample: Sample, prompt_ids: list[int]) -> None:
    response_len = random.randint(100, 300)
    response_ids = [random.randint(100, 150_000) for _ in range(response_len)]
    sample.status = Sample.Status.COMPLETED
    sample.tokens = prompt_ids + response_ids
    sample.loss_mask = [1] * response_len
    sample.response_length = response_len
    sample.reward = random.choice([0.0, 0.25, 0.5, 0.75, 1.0])
    sample.rollout_log_probs = None


def _build_aborted_sample(
    sample: Sample, tokenizer: Any, reward_value: float | None
) -> None:
    # Must have total_length > response_length (at least one "prompt" token)
    # to avoid a logit slicing bug in Megatron when this sample lands first
    # in a micro-batch.
    response_token = tokenizer.encode("failed")
    pad_id = tokenizer.pad_token_id or 0
    sample.status = Sample.Status.ABORTED
    sample.tokens = [pad_id] + response_token
    sample.loss_mask = [0] * len(response_token)
    sample.response_length = len(response_token)
    # reward=None is the key knob: exercises `_compute_zero_std_metrics`'s
    # round(None) path. The #989 fix buckets None under a "none" label.
    sample.reward = reward_value
    sample.rollout_log_probs = None


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_buffer: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Mock rollout — fast, no engines, no Harbor, no Daytona."""

    abort_frac = _env_float("MOCK_ROLLOUT_ABORT_FRAC", 0.5)
    use_none_reward = _env_flag("MOCK_ROLLOUT_NONE_REWARD", True)
    force_none_group = _env_flag("MOCK_ROLLOUT_FORCE_NONE_GROUP", False)
    sleep_secs = _env_float("MOCK_ROLLOUT_SLEEP_SECS", 0.5)

    samples_groups = data_buffer.get_samples(args.rollout_batch_size)
    state = GenerateState(args)
    tokenizer = state.tokenizer
    prompt_ids = tokenizer.encode("You are a helpful assistant solving a coding task.")

    t0 = time.time()
    completed = 0
    aborted = 0
    none_reward_count = 0

    aborted_reward = None if use_none_reward else 0.0

    for g_idx, group in enumerate(samples_groups):
        force_all_none = force_none_group and g_idx == 0
        for sample in group:
            if force_all_none or random.random() < abort_frac:
                _build_aborted_sample(sample, tokenizer, aborted_reward)
                aborted += 1
                if sample.reward is None:
                    none_reward_count += 1
            else:
                _build_completed_sample(sample, prompt_ids)
                completed += 1

            sample.metadata.update(
                {
                    "reward": sample.reward,
                    "trial_name": f"mock_{sample.metadata.get('instance_id', g_idx)}",
                    "task_name": sample.metadata.get("instance_id", f"mock_task_{g_idx}"),
                }
            )

    time.sleep(max(0.0, sleep_secs - (time.time() - t0)))
    elapsed = time.time() - t0

    n_total = completed + aborted

    def _safe_mean_reward() -> float:
        vals = [s.reward for g in samples_groups for s in g if s.reward is not None]
        return sum(vals) / max(len(vals), 1)

    metrics = {
        "mock_rollout/wall_time": elapsed,
        "mock_rollout/n_trials": n_total,
        "mock_rollout/n_completed": completed,
        "mock_rollout/n_aborted": aborted,
        "mock_rollout/n_none_reward": none_reward_count,
        "mock_rollout/mean_reward": _safe_mean_reward(),
        "mock_rollout/force_none_group": int(force_none_group),
        "mock_rollout/abort_frac": abort_frac,
        # Keep harbor/* keys too so downstream wandb panels work unchanged.
        "harbor/job_time": elapsed,
        "harbor/n_trials": n_total,
        "harbor/n_completed": completed,
        "harbor/n_truncated": 0,
        "harbor/n_failed": aborted,
        "harbor/mean_reward": _safe_mean_reward(),
        "harbor/synthetic": 1,
    }

    return RolloutFnTrainOutput(samples=samples_groups, metrics=metrics)


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Read reward from sample metadata; fall back to 0.0 for None."""
    reward = sample.metadata.get("reward")
    if reward is None:
        return 0.0
    return reward
