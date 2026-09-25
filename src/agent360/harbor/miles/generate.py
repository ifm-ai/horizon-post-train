# Modified for release: configurable deployment paths and public source references.
"""
Agent V2: reward, metrics, and rollout class.

The generate function is provided by:
    miles.rollout.generate_hub.agentic_tool_call.generate
with --custom-agent-function-path pointing to swe_agent_function.run

Task-type agnostic — reward is pre-computed by the Harbor environment
and stored in sample.metadata["reward"] regardless of task type.

Dynamic filter uses the general-purpose ``check_no_aborted`` from
``miles.rollout.filter_hub.dynamic_sampling_filters``.

Components:
  - reward_func: reads pre-computed reward from sample metadata
  - aggregate_agent_metrics: aggregates agent timing/count metrics
  - RolloutFn: InferenceRolloutFn subclass that logs agent metrics
"""

import json
import logging
import os
from collections import defaultdict

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.filter_hub.dynamic_sampling_filters import _flatten_samples
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample
from agent360.harbor.miles.utils import coerce_reward_to_float

logger = logging.getLogger(__name__)

# Per-task mean reward from a base-model pass@k run (last column of each jsonl
# row, keyed by task_id/instance_id), used as a cheap stand-in for "base model
# reward" so we don't have to run a second rollout pass every step. Set
# BASE_MODEL_REWARD_JSONL to enable reward/delta_vs_base_mean; unset -> skipped.
BASE_MODEL_REWARD_PATH = os.environ.get("BASE_MODEL_REWARD_JSONL", "")
_base_model_rewards: dict[str, float] | None = None


def _load_base_model_rewards() -> dict[str, float]:
    global _base_model_rewards
    if _base_model_rewards is not None:
        return _base_model_rewards
    rewards: dict[str, float] = {}
    if BASE_MODEL_REWARD_PATH:
        try:
            with open(BASE_MODEL_REWARD_PATH) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    rewards[row[0]] = float(row[-1])
        except OSError as exc:
            logger.warning("Could not load BASE_MODEL_REWARD_JSONL=%s: %s", BASE_MODEL_REWARD_PATH, exc)
    _base_model_rewards = rewards
    return rewards


# -- Reward --


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """Reward is pre-computed by the agent environment during generate().

    Handles both single-sample calls (from ``async_rm``) and batched calls
    (from ``batched_async_rm`` when ``--custom-rm-path`` is set).
    """
    if isinstance(samples, list):
        return [coerce_reward_to_float(s.metadata.get("reward")) for s in samples]
    return coerce_reward_to_float(samples.metadata.get("reward"))


# -- Reward logging (per-domain, unfiltered, pos/neg counts) --

# Pre-filter groups from the most recent rollout, stashed by log_all_samples so
# _call_train can compute unfiltered reward metrics. Rollouts run one at a time
# per trainer process, so a plain module global is safe.
# Key by rollout_id if rollouts ever overlap.
_last_all_samples: list[Sample] = []


def log_all_samples(args, all_samples, data_source=None):
    """Rollout all-samples hook: stash every generated group (pre-filter)."""
    global _last_all_samples
    _last_all_samples = list(_flatten_samples(all_samples))


def _take_all_samples() -> list[Sample]:
    """Return the stashed pre-filter samples and clear the stash."""
    global _last_all_samples
    taken, _last_all_samples = _last_all_samples, []
    return taken


def _domain(sample: Sample) -> str:
    tags = (sample.metadata or {}).get("tags") if hasattr(sample, "metadata") else None
    return tags[0] if tags else "untagged"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def reward_metrics(all_flat: list[Sample], trained_flat: list[Sample]) -> dict:
    """Per-domain / unfiltered reward and pos-neg counts for wandb.

    all_flat: every generated trajectory (before the dynamic filter dropped any).
    trained_flat: trajectories kept and not masked -- the effective train batch.
    """
    metrics = {}

    def rewards(samples):
        return [coerce_reward_to_float(s.metadata.get("reward")) for s in samples]

    for label, samples in [("unfiltered", all_flat), ("kept", trained_flat)]:
        rs = rewards(samples)
        if rs:
            metrics[f"reward/{label}_mean"] = _mean(rs)
        by_domain = defaultdict(list)
        for s, r in zip(samples, rs):
            by_domain[_domain(s)].append(r)
        for domain, drs in by_domain.items():
            metrics[f"reward/domain/{domain}_{label}_mean"] = _mean(drs)
            metrics[f"reward/domain/{domain}_{label}_count"] = len(drs)

    trained_rewards = rewards(trained_flat)
    pos = sum(1 for r in trained_rewards if r > 0)
    metrics["rollout/pos_trained"] = pos
    metrics["rollout/neg_trained"] = len(trained_rewards) - pos

    base_rewards = _load_base_model_rewards()
    if base_rewards:
        deltas = [
            r - base_rewards[iid]
            for s, r in zip(trained_flat, trained_rewards)
            if (iid := (s.metadata or {}).get("instance_id")) in base_rewards
        ]
        if deltas:
            metrics["reward/delta_vs_base_mean"] = _mean(deltas)

    return metrics


# -- Agent Metrics Aggregation --


def _collect_values(all_metrics: list[dict], key: str) -> list[float]:
    return [m[key] for m in all_metrics if key in m]


def _agg_mean(metrics: dict, all_metrics: list[dict], keys: list[str], prefix: str = "agent/", suffix: str = "_mean"):
    for key in keys:
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"{prefix}{key}{suffix}"] = sum(values) / len(values)


def aggregate_agent_metrics(samples: list[Sample]) -> dict:
    """Aggregate agent metrics across samples for logging."""
    all_metrics = [
        s.metadata.get("agent_metrics", {})
        for s in samples
        if hasattr(s, "metadata") and s.metadata and s.metadata.get("agent_metrics")
    ]
    if not all_metrics:
        return {}

    metrics = {}

    for key in ["turns", "tool_calls"]:
        values = _collect_values(all_metrics, key)
        if values:
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)
            metrics[f"agent/{key}_sum"] = sum(values)

    _agg_mean(metrics, all_metrics, ["model_query_time_sum", "env_execution_time_sum", "eval_time", "agent_run_time"])
    _agg_mean(metrics, all_metrics, ["time_per_turn", "model_query_time_avg", "env_execution_time_avg"], suffix="")
    _agg_mean(metrics, all_metrics, ["model_time_ratio", "env_time_ratio", "eval_time_ratio"], suffix="")

    values = _collect_values(all_metrics, "total_time")
    if values:
        metrics["agent/total_time_mean"] = sum(values) / len(values)
        metrics["agent/total_time_max"] = max(values)
        metrics["agent/total_time_min"] = min(values)

    for key in {k for m in all_metrics for k in m}:
        if f"agent/{key}_mean" in metrics or f"agent/{key}" in metrics:
            continue
        values = _collect_values(all_metrics, key)
        if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            metrics[f"agent/{key}_mean"] = sum(values) / len(values)

    return metrics


# -- Rollout Function --


class RolloutFn(InferenceRolloutFn):
    """Rollout function with agent metrics aggregation."""

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output = await super()._call_train(input)

        # -- TITO/R3 diagnostic (remove after validation) --
        for i, group in enumerate(output.samples):
            items = group if isinstance(group, list) else [group]
            for j, s in enumerate(items):
                if isinstance(s, list):
                    logger.info(f"DIAG sample[{i}][{j}]: nested list of {len(s)} items (multi-sample)")
                    continue
                re = getattr(s, "rollout_routed_experts", None)
                re_shape = re.shape if re is not None else None
                logger.info(
                    f"DIAG sample[{i}][{j}]: "
                    f"tokens={len(s.tokens) if s.tokens else 0}, "
                    f"resp_len={s.response_length}, "
                    f"logprobs={'len=' + str(len(s.rollout_log_probs)) if s.rollout_log_probs is not None else 'None'}, "
                    f"routed_experts={re_shape}, "
                    f"status={s.status}"
                )
        # -- end TITO/R3 diagnostic --

        all_samples = []
        for group in output.samples:
            if isinstance(group, list):
                all_samples.extend(group)
            else:
                all_samples.append(group)

        # Actual training batch after the dynamic filter dropped groups, plus how
        # much of it the rollout sample filter masked out (remove_sample -> no
        # gradient). samples_trained is the effective batch.
        flat = list(_flatten_samples(output.samples))
        n_masked = sum(1 for s in flat if getattr(s, "remove_sample", False))
        batch_metrics = {
            "rollout/groups_kept": len(output.samples),
            "rollout/samples_kept": len(flat),
            "rollout/samples_masked": n_masked,
            "rollout/samples_trained": len(flat) - n_masked,
        }

        trained_flat = [s for s in flat if not getattr(s, "remove_sample", False)]
        all_flat = _take_all_samples() or flat
        reward_stats = reward_metrics(all_flat, trained_flat)

        agent_metrics = aggregate_agent_metrics(all_samples)
        metrics = output.metrics or {}
        metrics.update(batch_metrics)
        metrics.update(reward_stats)
        metrics.update(agent_metrics)
        output.metrics = metrics
        logger.info(f"Rollout {input.rollout_id} batch metrics: {batch_metrics}")
        logger.info(f"Rollout {input.rollout_id} reward metrics: {reward_stats}")
        if agent_metrics:
            logger.info(f"Agent metrics for rollout {input.rollout_id}: {agent_metrics}")

        return output
