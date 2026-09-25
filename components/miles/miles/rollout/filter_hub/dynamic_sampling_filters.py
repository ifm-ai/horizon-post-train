# Modified for the RL360 coding overfit recipe: retain valid attempts in a group.
import torch

from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.utils.types import Sample

__all__ = [
    "check_reward_nonzero_std",
    "check_no_aborted",
    "check_no_infra_failures",
    "check_valid_reward_nonzero_std",
    "drop_zero_std_groups_and_extreme_pass_rate",
    "drop_truncated_or_extreme_pass_rate",
]

# Harbor exit_status values (from _extract_exit_status, after _TIMEOUT_EXCEPTION_MAP)
# for infra / non-policy failures: the rollout was severed by the environment,
# engine, or verifier rather than the policy, so it isn't valid training signal.
INFRA_FAILURE_EXIT_STATUSES = frozenset(
    {
        "AgentTimeout",
        "AgentTimeoutError",
        "HealthcheckError",
        "_K8sInternalInfraError",
        "Cancelled",
        "RewardFileNotFoundError",
        "AgentSetupTimeout",
        "AgentSetupTimeoutError",
        "VerifierTimeout",
        "VerifierTimeoutError",
        "VerifierCleanupError",
        "EnvStartTimeout",
        "EnvironmentStartTimeoutError",
        "TimeoutError",
        "AddTestsDirError",
        "NotFoundError",  # A lost inference session is an infrastructure failure.
    }
)


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    # Aborted samples never get a reward computed (see generate_and_rm in
    # sglang_rollout.py). Reject the group when any reward is None, matching
    # the check_no_aborted convention — a group with a missing reward can't
    # yield a reliable std and shouldn't contribute gradient signal.
    if any(r is None for r in rewards):
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    if len(rewards) < 2:
        raise ValueError(
            f"expected at least 2 samples per group, got {len(rewards)} — set --n-samples-per-prompt >= 2 for GRPO"
        )
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-8
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def _flatten_samples(samples):
    """Flatten samples that may contain nested lists (from --generate-multi-samples)."""
    for s in samples:
        if isinstance(s, list):
            yield from s
        else:
            yield s


def check_no_aborted(args, samples: list[Sample], **kwargs):
    """Reject entire group if any sample was aborted (e.g. env timeout, Docker crash)."""
    if any(s.status == Sample.Status.ABORTED for s in _flatten_samples(samples)):
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    return DynamicFilterOutput(keep=True)


def check_no_infra_failures(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Reject the group if any rollout aborted or failed for an infra reason,
    or if the group has zero reward std (all-correct / all-incorrect).

    Superset of ``check_no_aborted``: also drops samples whose Harbor
    ``exit_status`` is an infra / non-policy failure. Keyed on ``exit_status``
    because such rollouts may still record COMPLETED turns; aborted samples
    (zero records, no exit_status) are caught by the status check. Groups that
    survive the infra checks are then passed through ``check_reward_nonzero_std``
    so all-same-reward groups (advantage 0, no gradient) are dropped too.

        --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_infra_failures
    """
    flat_samples = list(_flatten_samples(samples))
    for sample in flat_samples:
        if sample.status == Sample.Status.ABORTED:
            return DynamicFilterOutput(keep=False, reason="group_has_aborted")
        exit_status = (sample.metadata or {}).get("exit_status", "")
        if exit_status in INFRA_FAILURE_EXIT_STATUSES:
            return DynamicFilterOutput(keep=False, reason=f"group_has_{exit_status}")
    return check_reward_nonzero_std(args, flat_samples, **kwargs)


def is_infra_failure(sample: Sample) -> bool:
    return (
        sample.status == Sample.Status.ABORTED
        or (sample.metadata or {}).get("exit_status", "") in INFRA_FAILURE_EXIT_STATUSES
        or (sample.metadata or {}).get("llm_judge_failed", False)
        or sample.reward is None
    )


def check_valid_reward_nonzero_std(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Keep groups with varying valid rewards despite individual infrastructure failures."""
    valid_samples = [sample for sample in _flatten_samples(samples) if not is_infra_failure(sample)]
    if len(valid_samples) < 2:
        return DynamicFilterOutput(keep=False, reason="insufficient_valid_rewards")
    return check_reward_nonzero_std(args, valid_samples, **kwargs)


def drop_zero_std_groups_and_extreme_pass_rate(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Filter groups with near-zero reward std or extreme mean rewards.
    For 0/1 rewards, mean reward is equivalent to pass rate --- so this function can be used to filter
    "easy" or "hard" groups

    Usage in config:
        --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.drop_zero_std_groups_and_extreme_pass_rate
        --dynamic-sampling-min-reward-std   (required, default: 1e-3)
        --dynamic-sampling-min-mean-reward  (required, default: 0.1)
        --dynamic-sampling-max-mean-reward  (required, default: 0.8)
    """
    flat_samples = list(_flatten_samples(samples))

    if not flat_samples:
        return DynamicFilterOutput(keep=False, reason="group_has_no_samples")

    rewards = [sample.get_reward_value(args) for sample in flat_samples]
    if any(reward is None for reward in rewards):
        return DynamicFilterOutput(keep=False, reason="group_has_missing_reward")

    if len(rewards) < 2:
        raise ValueError(
            f"expected at least 2 samples per group to check for standard deviation but got {len(rewards)} — set --n-samples-per-prompt >= 2 for GRPO"
        )
    reward_tensor = torch.tensor(rewards, dtype=torch.float64)
    mean_reward = reward_tensor.mean().item()
    std = reward_tensor.std().item()

    # get arguments for min_std, max_mean_reward, and min_mean_reward, see default values in `miles/utils/arguments.py` for reference
    min_std = getattr(args, "dynamic_sampling_min_reward_std", None)
    max_mean_reward = getattr(args, "dynamic_sampling_max_mean_reward", None)
    min_mean_reward = getattr(args, "dynamic_sampling_min_mean_reward", None)

    # check for none values
    if min_std is None:
        raise ValueError(
            "--dynamic-sampling-min-reward-std is required when using drop_zero_std_groups_and_extreme_pass_rate"
        )
    if max_mean_reward is None:
        raise ValueError(
            "--dynamic-sampling-max-mean-reward is required when using drop_zero_std_groups_and_extreme_pass_rate"
        )
    if min_mean_reward is None:
        raise ValueError(
            "--dynamic-sampling-min-mean-reward is required when using drop_zero_std_groups_and_extreme_pass_rate"
        )

    if std < min_std:
        return DynamicFilterOutput(keep=False, reason=f"near_zero_std_{min_std:g}")
    if mean_reward > max_mean_reward:
        return DynamicFilterOutput(keep=False, reason="mean_reward_too_high")
    if mean_reward < min_mean_reward:
        return DynamicFilterOutput(keep=False, reason="mean_reward_too_low")

    return DynamicFilterOutput(keep=True)


def drop_truncated_or_extreme_pass_rate(args, samples: list[Sample], **kwargs) -> DynamicFilterOutput:
    """Reject groups containing any truncated sample, then apply `drop_zero_std_groups_and_extreme_pass_rate` filter.

    Usage in config:
        --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.drop_truncated_or_extreme_pass_rate
        --dynamic-sampling-min-reward-std   (required, default: 1e-3)
        --dynamic-sampling-min-mean-reward  (required, default: 0.1)
        --dynamic-sampling-max-mean-reward  (required, default: 0.8)
    """
    flat_samples = list(_flatten_samples(samples))

    if any(sample.status == Sample.Status.TRUNCATED for sample in flat_samples):
        return DynamicFilterOutput(keep=False, reason="group_has_truncated")

    return drop_zero_std_groups_and_extreme_pass_rate(args, samples, **kwargs)
