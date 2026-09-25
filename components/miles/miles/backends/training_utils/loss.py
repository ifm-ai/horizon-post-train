import logging
import sys
from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from miles.utils.distributed_utils import distributed_masked_whiten
from miles.utils.misc import load_function
from miles.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_gspo_kl,
    compute_opd_reward,
    compute_opsm_mask,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from miles.utils.types import RolloutBatch, RolloutSamplingMask

from .cp_utils import (
    _allgather_cp_redistribute,
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
)
from .parallel import get_parallel_state

logger = logging.getLogger(__name__)


def _nan_dbg_flush_logs() -> None:
    """Flush logger/stdout/stderr so crash diagnostics survive abrupt failures."""
    for handler in logging.getLogger().handlers + logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        sys.stderr.flush()
    except Exception:
        pass


def _nan_dbg_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return -1


def _nan_dbg_scalar(value: float | int, device: torch.device) -> torch.Tensor:
    return torch.tensor(value, device=device, dtype=torch.float32)


def _nan_dbg_finite_stats(x: torch.Tensor) -> tuple[float, float, float, int]:
    x = x.detach()
    finite = torch.isfinite(x)
    bad = int((~finite).sum().item())
    if x.numel() == 0 or not bool(finite.any().item()):
        return float("nan"), float("nan"), float("nan"), bad
    xf = x[finite].float()
    return float(xf.min().item()), float(xf.max().item()), float(xf.abs().max().item()), bad


def _nan_dbg_batch_stats(batch: RolloutBatch) -> tuple[int, int, int, int]:
    response_len_max = max((int(x) for x in batch.get("response_lengths", [])), default=0)
    response_len_sum = sum(int(x) for x in batch.get("response_lengths", []))
    loss_tokens = [int(m.sum().item()) for m in batch.get("loss_masks", [])]
    loss_tokens_max = max(loss_tokens, default=0)
    loss_tokens_sum = sum(loss_tokens)
    return response_len_max, response_len_sum, loss_tokens_max, loss_tokens_sum


def _nan_dbg_warn_bad_tensor(name: str, x: torch.Tensor, *, batch: RolloutBatch, extra: str = "") -> None:
    x = x.detach()
    bad = int((~torch.isfinite(x)).sum().item())
    if bad == 0:
        return
    response_len_max, response_len_sum, loss_tokens_max, loss_tokens_sum = _nan_dbg_batch_stats(batch)
    logger.error(
        "NANDBG_BAD_TENSOR "
        f"rank={_nan_dbg_rank()} "
        f"name={name} "
        f"shape={tuple(x.shape)} "
        f"dtype={x.dtype} "
        f"nonfinite={bad} "
        f"response_len_max={response_len_max} "
        f"response_len_sum={response_len_sum} "
        f"loss_tokens_max={loss_tokens_max} "
        f"loss_tokens_sum={loss_tokens_sum} "
        f"{extra}"
    )
    _nan_dbg_flush_logs()


def _nan_dbg_warn_long_batch(args: Namespace, batch: RolloutBatch) -> None:
    response_len_max, response_len_sum, loss_tokens_max, loss_tokens_sum = _nan_dbg_batch_stats(batch)
    if response_len_max <= 65536 and loss_tokens_max <= 65536:
        return
    logger.warning(
        "NANDBG_LONG_BATCH "
        f"rank={_nan_dbg_rank()} "
        f"num_samples={len(batch.get('response_lengths', []))} "
        f"response_len_max={response_len_max} "
        f"response_len_sum={response_len_sum} "
        f"loss_tokens_max={loss_tokens_max} "
        f"loss_tokens_sum={loss_tokens_sum} "
        f"calculate_per_token_loss={args.calculate_per_token_loss} "
        f"loss_agg_mode={getattr(args, 'loss_agg_mode', None)} "
        f"use_dynamic_global_batch_size={args.use_dynamic_global_batch_size}"
    )
    _nan_dbg_flush_logs()


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]]:
    """Yield response logits, targets, and response-row spans per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Policy logits may stay in model precision; logprob code
            casts response chunks to fp32 after slicing to avoid full-logit
            fp32 materialization.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk, response_spans)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
        `response_spans` contains ordered half-open row ranges in the full response.
    """
    parallel_state = get_parallel_state()
    qkv_format = args.qkv_format

    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    logits = logits.div(args.rollout_temperature)

    cp_size = parallel_state.cp.size
    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
            else:
                end += total_length
                start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[total_length - response_length : total_length]
            response_spans = [(0, response_length)] if response_length else []
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = logits.size(0)
            cp_rank = parallel_state.cp.rank
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
                response_spans = []
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
                response_spans = [(s - logit_global_start, e - logit_global_start)]
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            prompt_length = total_length - response_length
            response_spans = [
                (start - prompt_length, stop - prompt_length) for start, stop in tokens_offset if start < stop
            ]

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        yield logits_chunk, tokens_chunk, response_spans


def _build_tp_sampling_mask(
    logits: torch.Tensor,
    sampling_mask: RolloutSamplingMask,
    response_spans: list[tuple[int, int]],
    tp_rank: int,
) -> torch.Tensor:
    mask = torch.zeros(logits.shape, dtype=torch.bool, device=logits.device)
    if not response_spans:
        return mask

    offsets = sampling_mask.offsets
    sizes = torch.cat([offsets[start + 1 : stop + 1] - offsets[start:stop] for start, stop in response_spans])
    parts = [sampling_mask.token_ids[offsets[start].item() : offsets[stop].item()] for start, stop in response_spans]
    token_ids = parts[0] if len(parts) == 1 else torch.cat(parts)

    vocab_size = logits.size(-1)
    vocab_start = tp_rank * vocab_size
    owned_entries = ((token_ids >= vocab_start) & (token_ids < vocab_start + vocab_size)).nonzero(as_tuple=True)[0]
    if owned_entries.numel() == 0:
        return mask

    rows = torch.repeat_interleave(sizes, output_size=token_ids.numel())[owned_entries]
    cols = token_ids[owned_entries].long() - vocab_start
    rows = rows.to(logits.device, non_blocking=True)
    cols = cols.to(logits.device, non_blocking=True)
    mask[rows, cols] = True
    return mask


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    rollout_sampling_masks: list[RolloutSamplingMask] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy values are always appended
    (even when `with_entropy=False`), but only included in the result dict
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `get_responses`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        non_loss_data: Unused; kept for API compatibility.
        rollout_sampling_masks: Fixed rollout support per full response; entropy remains unrestricted.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    parallel_state = get_parallel_state()
    assert non_loss_data

    log_probs_list = []
    entropy_list = []
    for i, (logits_chunk, tokens_chunk, response_spans) in enumerate(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )
    ):
        sampling_mask = None
        if rollout_sampling_masks is not None:
            sampling_mask = _build_tp_sampling_mask(
                logits_chunk, rollout_sampling_masks[i], response_spans, parallel_state.tp.rank
            )
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            parallel_state.tp.group,
            with_entropy=with_entropy,
            chunk_size=args.log_probs_chunk_size,
            true_on_policy=args.true_on_policy_mode,
            sampling_mask=sampling_mask,
        )

        log_probs_list.append(log_prob.reshape(-1))
        entropy_list.append(entropy)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    and "reinforce_plus_plus_baseline". When `args.normalize_advantages` is
    True, advantages are whitened across the data-parallel group using masked
    statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    parallel_state = get_parallel_state()
    log_probs: list[torch.Tensor] = rollout_data.get("rollout_log_probs" if args.use_rollout_logprobs else "log_probs")
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # return when not the last pp stage.
    if log_probs is None and values is None:
        return

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]

    if args.advantage_estimator in ["grpo", "gspo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "ppo":
        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = parallel_state.cp.rank
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, args.lambd
        )

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    elif args.advantage_estimator == "on_policy_distillation":
        student_log_probs = log_probs
        teacher_log_probs = rollout_data.get("teacher_log_probs")
        response_lengths = rollout_data.get("response_lengths")
        device = student_log_probs[0].device
        teacher_log_probs = [t_log_prob.to(device=device) for t_log_prob in teacher_log_probs]
        teacher_log_probs = [
            t_log_prob[-response_length:]
            for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
        ]
        advantages = [
            compute_opd_reward(student_log_prob, teacher_log_prob, getattr(args, "opd_reward_type", "logr"))
            for teacher_log_prob, student_log_prob in zip(teacher_log_probs, student_log_probs, strict=False)
        ]
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = parallel_state.cp.size
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len
                )

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = parallel_state.intra_dp.group

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis_delta = old_log_probs - rollout_log_probs
    tis = torch.exp(tis_delta)
    tis_abs = (tis - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    _, _, _, tis_bad = _nan_dbg_finite_stats(tis)
    if tis_bad:
        logger.error(
            "NANDBG_BAD_TIS "
            f"rank={_nan_dbg_rank()} "
            f"nonfinite={tis_bad} "
            f"tis_delta_min={_nan_dbg_finite_stats(tis_delta)[0]} "
            f"tis_delta_max={_nan_dbg_finite_stats(tis_delta)[1]} "
            f"tis_clip_low={args.tis_clip_low} "
            f"tis_clip={args.tis_clip}"
        )
        _nan_dbg_flush_logs()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    parallel_state = get_parallel_state()
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
        rollout_sampling_masks=batch.get("rollout_sampling_masks"),
    )

    log_probs = log_probs_and_entropy["log_probs"]

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    _nan_dbg_warn_bad_tensor("logits", logits, batch=batch)
    _nan_dbg_warn_bad_tensor("log_probs", log_probs, batch=batch)
    _nan_dbg_warn_bad_tensor("old_log_probs", old_log_probs, batch=batch)
    _nan_dbg_warn_bad_tensor("advantages", advantages, batch=batch)
    _nan_dbg_warn_bad_tensor("ppo_kl", ppo_kl, batch=batch)

    ratio_delta = -ppo_kl
    ratio = ratio_delta.exp()
    ratio_delta_min, ratio_delta_max, _, _ = _nan_dbg_finite_stats(ratio_delta)
    _nan_dbg_warn_bad_tensor(
        "ratio_exp_current_minus_old",
        ratio,
        batch=batch,
        extra=f"ratio_delta_min={ratio_delta_min} ratio_delta_max={ratio_delta_max}",
    )

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": batch["log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
            "parallel_state": parallel_state,
            "max_seq_lens": max_seq_lens,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with modified_response_masks for denominator correction
        # modified_response_masks will be sliced with cp in get_sum_of_sample_mean
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
            loss_agg_mode=getattr(args, "loss_agg_mode", None),
        )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    # Saved for per-domain fan-out (reducers below overwrite these names with scalars).
    _pg_loss_per_token = pg_loss
    _pg_clipfrac_per_token = pg_clipfrac
    _ppo_kl_per_token = ppo_kl

    log_probs_min, log_probs_max, _, log_probs_bad = _nan_dbg_finite_stats(log_probs)
    old_log_probs_min, old_log_probs_max, _, old_log_probs_bad = _nan_dbg_finite_stats(old_log_probs)
    _, _, advantage_absmax, advantages_bad = _nan_dbg_finite_stats(advantages)
    ppo_kl_min, ppo_kl_max, _, ppo_kl_bad = _nan_dbg_finite_stats(ppo_kl)
    _, ratio_max, _, ratio_bad = _nan_dbg_finite_stats(ratio)
    response_len_max, response_len_sum, loss_tokens_max, loss_tokens_sum = _nan_dbg_batch_stats(batch)
    tis_delta_min = float("nan")
    tis_delta_max = float("nan")
    tis_nonfinite_count = 0
    if args.get_mismatch_metrics or args.use_tis:
        rollout_log_probs_cat = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_tis_delta = old_log_probs - rollout_log_probs_cat
        tis_for_dbg = torch.exp(train_rollout_tis_delta)
        tis_delta_min, tis_delta_max, _, _ = _nan_dbg_finite_stats(train_rollout_tis_delta)
        _, _, _, tis_nonfinite_count = _nan_dbg_finite_stats(tis_for_dbg)

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # entropy loss
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    # Current and old policy log probs for policy_shift panel
    log_probs_metric = sum_of_sample_mean(log_probs).clone().detach()
    old_log_probs_metric = sum_of_sample_mean(old_log_probs).clone().detach()

    # Train-inference mismatch: compare inference engine vs FSDP at rollout time
    train_rollout_logprob_abs_diff = None
    train_rollout_logprob_diff = None
    _train_rollout_logprob_abs_per_token = None
    _train_rollout_logprob_signed_per_token = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs_cat = torch.cat(batch["rollout_log_probs"], dim=0)
        log_probs_batch_cat = torch.cat(batch["log_probs"], dim=0)
        _train_rollout_logprob_abs_per_token = (old_log_probs - rollout_log_probs_cat).abs()
        # signed: log π(inf) − log π(fsdp rollout)
        _train_rollout_logprob_signed_per_token = rollout_log_probs_cat - log_probs_batch_cat
        train_rollout_logprob_abs_diff = sum_of_sample_mean(_train_rollout_logprob_abs_per_token).clone().detach()
        train_rollout_logprob_diff = sum_of_sample_mean(_train_rollout_logprob_signed_per_token).clone().detach()

    # KL vs reference model — always log when ref present, regardless of use_kl_loss
    ref_kl_metric = None
    _ref_kl_per_token = None
    if "ref_log_probs" in batch and batch["ref_log_probs"]:
        ref_log_probs_cat = torch.cat(batch["ref_log_probs"], dim=0)
        _ref_kl_per_token = log_probs - ref_log_probs_cat
        ref_kl_metric = sum_of_sample_mean(_ref_kl_per_token).clone().detach()

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
        "log_probs": log_probs_metric,
        "old_log_probs": old_log_probs_metric,
        "nan_dbg/log_probs_min": _nan_dbg_scalar(log_probs_min, loss.device),
        "nan_dbg/log_probs_max": _nan_dbg_scalar(log_probs_max, loss.device),
        "nan_dbg/old_log_probs_min": _nan_dbg_scalar(old_log_probs_min, loss.device),
        "nan_dbg/old_log_probs_max": _nan_dbg_scalar(old_log_probs_max, loss.device),
        "nan_dbg/advantage_absmax": _nan_dbg_scalar(advantage_absmax, loss.device),
        "nan_dbg/ppo_kl_min": _nan_dbg_scalar(ppo_kl_min, loss.device),
        "nan_dbg/ppo_kl_max": _nan_dbg_scalar(ppo_kl_max, loss.device),
        "nan_dbg/ratio_max": _nan_dbg_scalar(ratio_max, loss.device),
        "nan_dbg/response_len_max": _nan_dbg_scalar(response_len_max, loss.device),
        "nan_dbg/response_len_sum": _nan_dbg_scalar(response_len_sum, loss.device),
        "nan_dbg/loss_tokens_max": _nan_dbg_scalar(loss_tokens_max, loss.device),
        "nan_dbg/loss_tokens_sum": _nan_dbg_scalar(loss_tokens_sum, loss.device),
        "nan_dbg/nonfinite_count": _nan_dbg_scalar(
            log_probs_bad + old_log_probs_bad + advantages_bad + ppo_kl_bad + ratio_bad,
            loss.device,
        ),
        "nan_dbg/tis_delta_min": _nan_dbg_scalar(tis_delta_min, loss.device),
        "nan_dbg/tis_delta_max": _nan_dbg_scalar(tis_delta_max, loss.device),
        "nan_dbg/tis_nonfinite_count": _nan_dbg_scalar(tis_nonfinite_count, loss.device),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff
        reported_loss["train_rollout_logprob_diff"] = train_rollout_logprob_diff

    if ref_kl_metric is not None:
        reported_loss["ref_kl"] = ref_kl_metric

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # Per-domain fan-out: activated by batch["domains"] (set when samples carry
    # metadata["domain"]). batch["all_domains"] is cached on DataIterator so every
    # microbatch emits the same key set (aggregate_train_losses keys positionally).
    # grad_norm isn't split: backward() has already mixed gradients.
    if batch.get("domains") and batch.get("all_domains"):
        per_token = {
            "log_probs": log_probs,
            "old_log_probs": old_log_probs,
            "pg_loss": _pg_loss_per_token,
            "pg_clipfrac": _pg_clipfrac_per_token,
            "ppo_kl": _ppo_kl_per_token,
            "entropy_loss": entropy,
        }
        if _ref_kl_per_token is not None:
            per_token["ref_kl"] = _ref_kl_per_token
        if _train_rollout_logprob_signed_per_token is not None:
            per_token["train_rollout_logprob_diff"] = _train_rollout_logprob_signed_per_token
            per_token["train_rollout_logprob_abs_diff"] = _train_rollout_logprob_abs_per_token
        if args.get_mismatch_metrics or args.use_tis:
            per_token["ois"] = ois
            per_token.update(tis_metrics)

        for d in batch["all_domains"]:
            masked = [
                lm if dd == d else torch.zeros_like(lm)
                for dd, lm in zip(batch["domains"], batch["loss_masks"], strict=False)
            ]
            reducer = get_sum_of_sample_mean(
                total_lengths,
                response_lengths,
                masked,
                args.calculate_per_token_loss,
                args.qkv_format,
                max_seq_lens,
                loss_agg_mode=getattr(args, "loss_agg_mode", None),
            )
            for name, t in per_token.items():
                reported_loss[f"{name}/{d}"] = reducer(t).clone().detach()
            reported_loss[f"loss/{d}"] = (
                reported_loss[f"pg_loss/{d}"] - args.entropy_coef * reported_loss[f"entropy_loss/{d}"]
            )

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
    apply_megatron_loss_scaling: bool = False,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    parallel_state = get_parallel_state()
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])
    num_samples = len(batch["response_lengths"])

    if args.loss_type == "policy_loss":
        _nan_dbg_warn_long_batch(args, batch)

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
        loss_agg_mode=getattr(args, "loss_agg_mode", None),
    )

    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(
            func,
            args,
            batch,
            logits,
            sum_of_sample_mean,
        )
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # With allgather-CP, some CP ranks may have no loss-contributing tokens (e.g., all
    # padding). Without this, gradient doesn't flow through their attention path, so
    # the CP gather's backward (reduce-scatter) is not called, deadlocking other CP
    # ranks that call it. Adding this zero loss forces autograd to traverse the full
    # graph on every rank without changing gradient values.
    if parallel_state.cp.size > 1 and args.allgather_cp:
        loss = loss + 0 * logits.sum()

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    assert args.use_dynamic_global_batch_size == ("dynamic_global_batch_size" in batch)

    # Resolve effective loss mode: --loss-agg-mode takes precedence over --calculate-per-token-loss
    loss_agg_mode = getattr(args, "loss_agg_mode", None)
    if loss_agg_mode is None:
        loss_agg_mode = "token-sum" if args.calculate_per_token_loss else "sample-mean"
    uses_token_normalization = loss_agg_mode in ("token-mean", "token-sum")

    # Scale loss for distributed training.
    # - sample-mean: divide by global_batch_size (number of samples)
    # - token-mean/token-sum: use num_tokens as normalizer
    #   For Megatron: the external normalizer handles division by num_tokens.
    #   For FSDP (apply_megatron_loss_scaling=False): token-mean explicitly
    #   divides by num_tokens here since FSDP ignores the returned normalizer.
    global_batch_size = batch.get("dynamic_global_batch_size", args.global_batch_size)
    if not uses_token_normalization:
        # sample-mean path
        if apply_megatron_loss_scaling:
            loss = loss * num_microbatches / global_batch_size * parallel_state.intra_dp_cp.size
        else:
            loss = loss / global_batch_size * parallel_state.intra_dp.size
    else:
        if apply_megatron_loss_scaling:
            # Megatron normalizes externally via num_tokens normalizer
            loss = loss * parallel_state.cp.size
        elif loss_agg_mode == "token-mean":
            # FSDP: normalize by num_tokens explicitly (FSDP ignores the normalizer)
            loss = loss / torch.clamp_min(num_tokens, 1) * parallel_state.intra_dp.size
        # token-sum on FSDP: no normalization (raw sum, legacy behavior)

    return (
        loss,
        torch.tensor(num_tokens if uses_token_normalization else 1, device=logits.device),
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [
                    num_samples if not uses_token_normalization else num_tokens,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
