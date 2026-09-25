"""Token-mean pg_loss reducer for miles.

Implements the "token-mean" loss aggregation mode:
    loss = masked_sum(all token losses) / total_masked_tokens * dp_size

Every token contributes equally to the gradient regardless of which sequence
it belongs to. This differs from miles' default (sample-mean) where each
*sequence* contributes equally, giving shorter sequences disproportionate
per-token weight.

Compatible with context parallelism (CP > 1) by replicating the same
CP-aware mask chunking that the built-in reducer uses.

Assumptions:
    - qkv_format is "thd" (the default). The custom reducer API doesn't expose
      qkv_format, so we hardcode it. If you use --qkv-format bshd, CP offsets
      will be wrong — either switch to CP=1 or update this code.
    - max_seq_lens is None (no variable-length batching). Same limitation as above.
    - These match the framework defaults.

Usage:
    --custom-pg-loss-reducer-function-path miles360.token_mean_reducer.get_token_mean_reducer
"""

from collections.abc import Callable

import torch
from megatron.core import mpu


def get_token_mean_reducer(
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    calculate_per_token_loss: bool = False,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a reducer that computes token-mean over the batch.

    Formula: masked_sum(losses) / total_masked_tokens * dp_size

    The dp_size factor ensures gradient magnitudes stay consistent across
    different data-parallel configurations.
    total_mask_tokens uses the full (un-chunked) masks — this is correct because
    CP ranks process chunks of the same micro-batch and gradients are summed
    across CP ranks via all-reduce, reconstructing the full masked sum.
    """
    dp_size = mpu.get_data_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    total_mask_tokens = sum(lm.sum() for lm in loss_masks)

    if cp_size == 1:
        def token_mean(x: torch.Tensor) -> torch.Tensor:
            raw = sum(
                (x_i * lm).sum()
                for x_i, lm in zip(x.split(response_lengths, dim=0), loss_masks, strict=True)
            )
            return raw / torch.clamp_min(total_mask_tokens, 1) * dp_size
    else:
        # CP > 1: the tensor x is chunked but loss_masks are full.
        # Replicate miles' internal CP chunking to produce masks that match x.
        # See miles/miles/backends/training_utils/cp_utils.py:get_sum_of_sample_mean
        from miles.backends.training_utils.cp_utils import get_logits_and_tokens_offset_with_cp

        cp_chunk_lengths = []
        chunked_loss_masks = []

        for total_length, response_length, loss_mask in zip(
            total_lengths, response_lengths, loss_masks, strict=True
        ):
            prompt_length = total_length - response_length
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, "thd", None
            )
            assert len(tokens_offset) == 2, (
                f"token_mean_reducer expects exactly 2 CP chunks (thd format), got {len(tokens_offset)}. "
                f"If miles changed CP chunking, this reducer needs updating."
            )
            assert tokens_offset[0][0] >= prompt_length, (
                f"tokens_offset[0][0]={tokens_offset[0][0]} < prompt_length={prompt_length}, "
                f"negative slice index would produce wrong masks"
            )
            assert tokens_offset[1][0] >= prompt_length, (
                f"tokens_offset[1][0]={tokens_offset[1][0]} < prompt_length={prompt_length}, "
                f"negative slice index would produce wrong masks"
            )
            lm_0 = loss_mask[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
            lm_1 = loss_mask[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
            chunked = torch.cat([lm_0, lm_1], dim=0)
            chunked_loss_masks.append(chunked)
            cp_chunk_lengths.append(chunked.size(0))

        def token_mean(x: torch.Tensor) -> torch.Tensor:
            expected = sum(cp_chunk_lengths)
            assert x.size(0) == expected, (
                f"token_mean_reducer: tensor size {x.size(0)} != sum(cp_chunk_lengths) {expected}. "
                f"This likely means qkv_format is not 'thd' or max_seq_lens is set."
            )
            raw = sum(
                (x_i * clm).sum()
                for x_i, clm in zip(x.split(cp_chunk_lengths, dim=0), chunked_loss_masks, strict=True)
            )
            return raw / torch.clamp_min(total_mask_tokens, 1) * dp_size

    return token_mean
