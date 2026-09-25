# Modified for the RL360 coding overfit recipe: mask failed infrastructure attempts.
from copy import deepcopy

from miles.rollout.filter_hub.dynamic_sampling_filters import _flatten_samples, is_infra_failure
from miles.utils.types import Sample

__all__ = ["mask_truncated", "mask_truncated_and_llm_judge_failed", "mask_token_truncated", "mask_infra_failures"]


def mask_infra_failures(args, samples: list[Sample]) -> None:
    """Mask failed infrastructure attempts; retain policy failures and truncations.

    Empty aborted attempts borrow a valid sequence solely to satisfy tensor
    shape requirements. Their loss mask is zero, so they contribute no gradient.
    """
    flat_samples = list(_flatten_samples(samples))
    template = next(
        (sample for sample in flat_samples if not is_infra_failure(sample) and sample.response_length > 0),
        None,
    )
    for sample in flat_samples:
        if not is_infra_failure(sample):
            continue
        sample.remove_sample = True
        if sample.reward is None:
            sample.reward = {args.reward_key: 0.0} if args.reward_key else 0.0
        if not sample.tokens or sample.response_length == 0:
            if template is None:
                raise ValueError("Cannot pad aborted attempts without a valid training sequence")
            for field in (
                "tokens", "response", "response_length", "rollout_log_probs",
                "rollout_routed_experts", "weight_versions", "multimodal_inputs",
                "multimodal_train_inputs",
            ):
                setattr(sample, field, deepcopy(getattr(template, field)))
        sample.loss_mask = [0] * sample.response_length

# Harbor exit_status values for rollouts truncated by running out of output
# tokens or context. Real attempts (verifier reward is meaningful) but the
# trajectory shouldn't get gradient — masked, reward kept in the group baseline.
TRUNCATION_EXIT_STATUSES = frozenset(
    {
        "BadRequestError",
        "ContextWindowExceededError",
        "OutputLengthExceededError",
    }
)


def mask_token_truncated(args, samples: list[Sample]) -> None:
    """Mask token/context-truncated samples (zero gradient, reward kept in baseline).

    Keyed on Harbor ``exit_status`` (BadRequest/ContextWindow record COMPLETED
    turns, so a status check misses them); also covers TRUNCATED status.

        --rollout-sample-filter-path miles.rollout.filter_hub.rollout_filters.mask_token_truncated
    """
    for sample in _flatten_samples(samples):
        # note that the following will cause a crash with `--generate-multi-samples` but that's intentionally not supported by us
        exit_status = (sample.metadata or {}).get("exit_status", "")
        if exit_status in TRUNCATION_EXIT_STATUSES or sample.status == Sample.Status.TRUNCATED:
            sample.remove_sample = True


def mask_truncated(args, samples: list[Sample]) -> None:
    """Mask truncated samples so they are excluded from training.

    Usage in config:
        --rollout-sample-filter-path miles.rollout.filter_hub.rollout_filters.mask_truncated
    """
    for sample in _flatten_samples(samples):
        if sample.status == Sample.Status.TRUNCATED:
            sample.remove_sample = True


def mask_truncated_and_llm_judge_failed(args, samples: list[Sample]) -> None:
    """Mask truncated samples and samples where the LLM judge failed so they are excluded from training.

    Usage in config:
        --rollout-sample-filter-path miles.rollout.filter_hub.rollout_filters.mask_truncated_and_llm_judge_failed

    Requires the reward function to return ``llm_judge_failed: True`` in its score dict when judge
    calls fail (stored in ``sample.metadata["llm_judge_failed"]`` by async_rm).
    """
    for sample in _flatten_samples(samples):
        if sample.status == Sample.Status.TRUNCATED or sample.metadata.get("llm_judge_failed"):
            sample.remove_sample = True
