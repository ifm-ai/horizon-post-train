from copy import copy
from dataclasses import fields

from miles.utils.types import RolloutSamplingMask, Sample


def drop_samples_after_first_non_completed(samples: list[Sample]) -> tuple[list[Sample], int]:
    """Keep turns up to and including the first non-COMPLETED one.

    A turn that ended early (engine abort during rollout shutdown, per-turn
    length limit) means every later turn was conditioned on incomplete
    output, so those turns are invalid training data. Dropping them
    establishes the ``merge_samples`` invariant that only the final sample
    may be non-COMPLETED.

    Returns the kept prefix and the number of dropped samples.
    """
    for i, sample in enumerate(samples[:-1]):
        if sample.status != Sample.Status.COMPLETED:
            return samples[: i + 1], len(samples) - i - 1
    return samples, 0


def merge_samples(samples: list[Sample], tokenizer) -> Sample:
    acc = samples[0]
    # Concatenate masks once after all turns.
    mask_parts = []
    if len(samples) > 1 and acc.rollout_sampling_mask is not None:
        mask_parts.append(acc.rollout_sampling_mask)
    for sample in samples[1:]:
        if mask_parts:
            obs_tokens = sample.tokens[len(acc.tokens) : len(sample.tokens) - sample.response_length]
            mask_parts.extend((RolloutSamplingMask.singletons(obs_tokens), sample.rollout_sampling_mask))
        acc = _merge_sample_pair(acc, sample, tokenizer=tokenizer)
    if mask_parts:
        acc.rollout_sampling_mask = RolloutSamplingMask.concatenate(*mask_parts)
    return acc


def _merge_sample_pair(a: Sample, b: Sample, tokenizer) -> Sample:
    """Merge two samples generated from sibling inference engine calls."""
    a, b = copy(a), copy(b)

    def _merge_equal_value(field):
        x = getattr(a, field)
        y = getattr(b, field)
        assert x == y, f"{field} mismatch: a.{field}={x}, b.{field}={y}"
        return x

    def _fill_defaults(sample: Sample):
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = [0.0] * sample.response_length

    _fill_defaults(a)
    _fill_defaults(b)

    obs_len = len(b.tokens) - len(a.tokens) - b.response_length
    obs_tokens = b.tokens[len(a.tokens) : len(a.tokens) + obs_len]

    response_decoded = _sample_response_decoded(a) and _sample_response_decoded(b)
    obs_text = tokenizer.decode(obs_tokens) if response_decoded else ""
    response = a.response + obs_text + b.response if response_decoded else ""
    metadata = _merge_metadata(a.metadata, b.metadata, response_decoded=response_decoded)

    try:
        a.validate()
        b.validate()
        assert _startswith(short=a.prompt, long=b.prompt), "b.prompt must start with a.prompt"
        assert _startswith(short=a.tokens, long=b.tokens), "b.tokens must start with a.tokens"
        assert obs_len > 0, f"obs_len must be > 0, got {obs_len}"
        if a.rollout_routed_experts is not None:
            assert a.rollout_routed_experts.shape[0] <= b.rollout_routed_experts.shape[0]
        assert a.status == Sample.Status.COMPLETED, f"a.status must be COMPLETED, got {a.status}"

        return _create_with_all_fields(
            Sample,
            group_index=_merge_equal_value("group_index"),
            index=_merge_equal_value("index"),
            prompt=b.prompt,
            tokens=b.tokens,
            multimodal_inputs=_merge_equal_value("multimodal_inputs"),
            multimodal_train_inputs=_merge_equal_value("multimodal_train_inputs"),
            response=response,
            response_length=a.response_length + obs_len + b.response_length,
            label=_merge_equal_value("label"),
            reward=_merge_equal_value("reward"),
            loss_mask=a.loss_mask + [0] * obs_len + b.loss_mask,
            weight_versions=a.weight_versions + b.weight_versions,
            rollout_log_probs=a.rollout_log_probs + [0.0] * obs_len + b.rollout_log_probs,
            rollout_sampling_mask=None,
            rollout_routed_experts=b.rollout_routed_experts,
            remove_sample=_merge_equal_value("remove_sample"),
            status=b.status,
            metadata=metadata,
            generate_function_path=_merge_equal_value("generate_function_path"),
            train_metadata=_merge_equal_value("train_metadata"),
            session_id=_merge_equal_value("session_id"),
            non_generation_time=_merge_equal_value("non_generation_time"),
            spec_info=_merge_spec_info(a.spec_info, b.spec_info),
            prefix_cache_info=_merge_prefix_cache_info(a.prefix_cache_info, b.prefix_cache_info),
        )
    except AssertionError as e:
        e.add_note(f"{a=} {b=}")
        raise


def _sample_response_decoded(sample: Sample) -> bool:
    """Old samples without the metadata flag are treated as decoded."""
    metadata = getattr(sample, "metadata", None)
    if metadata is None:
        return True
    return metadata.get("response_decoded", True)


def _merge_metadata(
    a: dict | None,
    b: dict | None,
    *,
    response_decoded: bool,
) -> dict:
    a = {} if a is None else dict(a)
    b = {} if b is None else dict(b)

    a_without_response_decoded = dict(a)
    b_without_response_decoded = dict(b)
    a_without_response_decoded.pop("response_decoded", None)
    b_without_response_decoded.pop("response_decoded", None)

    assert (
        a_without_response_decoded == b_without_response_decoded
    ), f"metadata mismatch: a.metadata={a}, b.metadata={b}"

    merged = dict(a_without_response_decoded)
    merged["response_decoded"] = response_decoded
    return merged


def _merge_spec_info(a: Sample.SpecInfo, b: Sample.SpecInfo) -> Sample.SpecInfo:
    def _merge_plus_value(field):
        return getattr(a, field) + getattr(b, field)

    return _create_with_all_fields(
        Sample.SpecInfo,
        spec_accept_token_num=_merge_plus_value("spec_accept_token_num"),
        spec_draft_token_num=_merge_plus_value("spec_draft_token_num"),
        spec_verify_ct=_merge_plus_value("spec_verify_ct"),
        completion_token_num=_merge_plus_value("completion_token_num"),
    )


def _merge_prefix_cache_info(a: Sample.PrefixCacheInfo, b: Sample.PrefixCacheInfo) -> Sample.PrefixCacheInfo:
    def _merge_plus_value(field):
        return getattr(a, field) + getattr(b, field)

    return _create_with_all_fields(
        Sample.PrefixCacheInfo,
        cached_tokens=_merge_plus_value("cached_tokens"),
        total_prompt_tokens=_merge_plus_value("total_prompt_tokens"),
    )


def _create_with_all_fields(cls, **kwargs):
    expected = {f.name for f in fields(cls)}
    actual = set(kwargs.keys())
    assert (
        expected == actual
    ), f"{cls.__name__} field mismatch. Missing: {expected - actual}, Extra: {actual - expected}"
    return cls(**kwargs)


def _startswith(*, short, long) -> bool:
    if isinstance(short, str) and isinstance(long, str):
        return long.startswith(short)
    if isinstance(short, list) and isinstance(long, list):
        return (len(long) >= len(short)) and (long[: len(short)] == short)
    raise NotImplementedError
