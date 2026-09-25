# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Checkpoint mapping from native xLLM MoVA tensors to Megatron Core.

The source state dict is expected to contain full tensors after merging xLLM's
model-parallel shards.  This module owns only tensor semantics and local-rank
slicing; source checkpoint I/O and Megatron distributed-checkpoint saving stay
in the conversion tool.
"""

from typing import Mapping

import torch
from torch import Tensor

from megatron.core.transformer.moe.experts import SequentialMLP
from megatron.core.transformer.mova import GroupedGemmMoVAValueExperts, SequentialMoVAValueExperts
from megatron.core.utils import get_pg_rank, get_pg_size


def _require(source: Mapping[str, Tensor], key: str) -> Tensor:
    try:
        return source[key]
    except KeyError as error:
        raise KeyError(f"Missing required xLLM tensor: {key}") from error


@torch.no_grad()
def _copy(target: Tensor, source: Tensor, key: str) -> None:
    if target.shape != source.shape:
        raise ValueError(
            f"Shape mismatch for {key}: source={tuple(source.shape)}, "
            f"target={tuple(target.shape)}"
        )
    target.copy_(source.to(device=target.device, dtype=target.dtype))


def _tp_slice(tensor: Tensor, axis: int, rank: int, size: int, key: str) -> Tensor:
    if tensor.shape[axis] % size:
        raise ValueError(
            f"Cannot shard {key} axis {axis} of size {tensor.shape[axis]} over TP={size}"
        )
    return tensor.chunk(size, dim=axis)[rank].contiguous()


def _pack_gqa_projection(
    query: Tensor,
    gate: Tensor,
    key: Tensor,
    value: Tensor | None,
    *,
    num_query_groups: int,
    key_name: str,
) -> Tensor:
    """Pack xLLM head-major projections into MCore's per-GQA-group layout."""

    hidden_size = query.shape[1]
    if query.shape != gate.shape:
        raise ValueError(f"{key_name}: query and gate shapes differ")
    if query.shape[0] % num_query_groups:
        raise ValueError(f"{key_name}: query width is not divisible by query groups")
    if key.ndim != 2 or key.shape[1] != hidden_size:
        raise ValueError(f"{key_name}: invalid key projection shape {tuple(key.shape)}")
    if key.shape[0] % num_query_groups:
        raise ValueError(f"{key_name}: invalid key projection shape {tuple(key.shape)}")
    if value is not None and value.shape != key.shape:
        raise ValueError(f"{key_name}: key and value shapes differ")

    query = query.view(num_query_groups, -1, hidden_size)
    gate = gate.view(num_query_groups, -1, hidden_size)
    key = key.view(num_query_groups, -1, hidden_size)
    projections = [query, gate, key]
    if value is not None:
        projections.append(value.view(num_query_groups, -1, hidden_size))
    return torch.cat(projections, dim=1).reshape(-1, hidden_size).contiguous()


def _load_dense_attention(layer, source: Mapping[str, Tensor], layer_idx: int) -> None:
    prefix = f"layers.{layer_idx}.attention"
    attention = layer.self_attention
    tp_group = attention.pg_collection.tp
    tp_rank, tp_size = get_pg_rank(tp_group), get_pg_size(tp_group)
    packed = _pack_gqa_projection(
        _require(source, f"{prefix}.wq.weight"),
        _require(source, f"{prefix}.wr.weight"),
        _require(source, f"{prefix}.wk.weight"),
        _require(source, f"{prefix}.wv.weight"),
        num_query_groups=attention.config.num_query_groups,
        key_name=f"{prefix}.linear_qkv",
    )
    _copy(
        attention.linear_qkv.weight,
        _tp_slice(packed, 0, tp_rank, tp_size, f"{prefix}.linear_qkv"),
        f"{prefix}.linear_qkv",
    )
    output = _require(source, f"{prefix}.wo.weight")
    _copy(
        attention.linear_proj.weight,
        _tp_slice(output, 1, tp_rank, tp_size, f"{prefix}.wo.weight"),
        f"{prefix}.wo.weight",
    )


def _load_mova_value_experts(attention, source_weight: Tensor, key: str) -> None:
    value_experts = attention.value_projection.experts
    tp_group = value_experts.tp_group
    tp_rank, tp_size = get_pg_rank(tp_group), get_pg_size(tp_group)
    if source_weight.ndim != 3:
        raise ValueError(f"{key} must have shape [experts, value_width, hidden]")

    if isinstance(value_experts, SequentialMoVAValueExperts):
        if len(value_experts.experts) != source_weight.shape[0]:
            raise ValueError(f"{key}: source and target value-expert counts differ")
        for expert_idx, expert in enumerate(value_experts.experts):
            local_weight = _tp_slice(source_weight[expert_idx], 0, tp_rank, tp_size, key)
            _copy(expert.weight, local_weight, f"{key}[{expert_idx}]")
        return

    if isinstance(value_experts, GroupedGemmMoVAValueExperts):
        # Production grouped GEMM consumes [expert, hidden_per_tp, output].
        local_weight = _tp_slice(source_weight, 2, tp_rank, tp_size, key)
        _copy(value_experts.weight, local_weight.transpose(1, 2), key)
        return

    raise TypeError(f"Unsupported MoVA value backend: {type(value_experts).__name__}")


def _load_mova_attention(layer, source: Mapping[str, Tensor], layer_idx: int) -> None:
    prefix = f"layers.{layer_idx}.mova"
    attention = layer.self_attention
    tp_group = attention.pg_collection.tp
    tp_rank, tp_size = get_pg_rank(tp_group), get_pg_size(tp_group)
    packed = _pack_gqa_projection(
        _require(source, f"{prefix}.wq.weight"),
        _require(source, f"{prefix}.wr.weight"),
        _require(source, f"{prefix}.wk.weight"),
        None,
        num_query_groups=attention.config.num_query_groups,
        key_name=f"{prefix}.linear_qkg",
    )
    _copy(
        attention.linear_qkg.weight,
        _tp_slice(packed, 0, tp_rank, tp_size, f"{prefix}.linear_qkg"),
        f"{prefix}.linear_qkg",
    )
    output = _require(source, f"{prefix}.wo.weight")
    _copy(
        attention.linear_proj.weight,
        _tp_slice(output, 1, tp_rank, tp_size, f"{prefix}.wo.weight"),
        f"{prefix}.wo.weight",
    )
    _copy(
        attention.value_projection.router.weight,
        _require(source, f"{prefix}.router.weight"),
        f"{prefix}.router.weight",
    )
    if attention.value_projection.router.expert_bias is not None:
        _copy(
            attention.value_projection.router.expert_bias,
            _require(source, f"{prefix}.router.bias"),
            f"{prefix}.router.bias",
        )
    _load_mova_value_experts(
        attention, _require(source, f"{prefix}.wv.weight"), f"{prefix}.wv.weight"
    )


def _pack_swiglu(gate: Tensor, up: Tensor, key: str) -> Tensor:
    if gate.shape != up.shape:
        raise ValueError(f"{key}: SwiGLU gate and up projections differ")
    return torch.cat((gate, up), dim=0).contiguous()


def _load_dense_mlp(layer, source: Mapping[str, Tensor], layer_idx: int) -> None:
    prefix = f"layers.{layer_idx}.nffn"
    mlp = layer.mlp
    tp_group = mlp.tp_group
    tp_rank, tp_size = get_pg_rank(tp_group), get_pg_size(tp_group)
    packed = _pack_swiglu(
        _require(source, f"{prefix}.fc1.weight"),
        _require(source, f"{prefix}.fc3.weight"),
        f"{prefix}.linear_fc1",
    )
    _copy(
        mlp.linear_fc1.weight,
        _tp_slice(packed, 0, tp_rank, tp_size, f"{prefix}.linear_fc1"),
        f"{prefix}.linear_fc1",
    )
    down = _require(source, f"{prefix}.fc2.weight")
    _copy(
        mlp.linear_fc2.weight,
        _tp_slice(down, 1, tp_rank, tp_size, f"{prefix}.fc2.weight"),
        f"{prefix}.fc2.weight",
    )


def _load_sparse_mlp(layer, source: Mapping[str, Tensor], layer_idx: int) -> None:
    prefix = f"layers.{layer_idx}.moe"
    moe = layer.mlp
    _copy(moe.router.weight, _require(source, f"{prefix}.router.weight"), f"{prefix}.router.weight")
    if moe.router.expert_bias is not None:
        _copy(
            moe.router.expert_bias,
            _require(source, f"{prefix}.router.bias"),
            f"{prefix}.router.bias",
        )

    if not isinstance(moe.experts, SequentialMLP):
        raise TypeError(
            "xLLM conversion must build FFN experts with SequentialMLP; "
            "the resulting distributed checkpoint is load-compatible with grouped MLP backends"
        )
    weight1 = _require(source, f"{prefix}.experts.weight1")
    weight2 = _require(source, f"{prefix}.experts.weight2")
    weight3 = _require(source, f"{prefix}.experts.weight3")
    ep_rank, ep_size = get_pg_rank(moe.experts.ep_group), get_pg_size(moe.experts.ep_group)
    expert_tp_rank = get_pg_rank(moe.experts.tp_group)
    expert_tp_size = get_pg_size(moe.experts.tp_group)
    if weight1.shape[0] % ep_size:
        raise ValueError(f"{prefix}: expert count is not divisible by EP={ep_size}")
    local_expert_count = weight1.shape[0] // ep_size
    expert_offset = ep_rank * local_expert_count
    if len(moe.experts.local_experts) != local_expert_count:
        raise ValueError(f"{prefix}: source and target local expert counts differ")
    for local_idx, expert in enumerate(moe.experts.local_experts):
        global_idx = expert_offset + local_idx
        packed = _pack_swiglu(
            weight1[global_idx], weight3[global_idx], f"{prefix}.expert[{global_idx}]"
        )
        _copy(
            expert.linear_fc1.weight,
            _tp_slice(
                packed,
                0,
                expert_tp_rank,
                expert_tp_size,
                f"{prefix}.expert[{global_idx}].linear_fc1",
            ),
            f"{prefix}.expert[{global_idx}].linear_fc1",
        )
        _copy(
            expert.linear_fc2.weight,
            _tp_slice(
                weight2[global_idx],
                1,
                expert_tp_rank,
                expert_tp_size,
                f"{prefix}.expert[{global_idx}].linear_fc2",
            ),
            f"{prefix}.expert[{global_idx}].linear_fc2",
        )

    shared = moe.shared_experts
    if shared is None:
        raise ValueError(f"{prefix}: target is missing the xLLM shared expert")
    tp_group = shared.tp_group
    tp_rank, tp_size = get_pg_rank(tp_group), get_pg_size(tp_group)
    shared_packed = _pack_swiglu(
        _require(source, f"{prefix}.fc1.weight"),
        _require(source, f"{prefix}.fc3.weight"),
        f"{prefix}.shared.linear_fc1",
    )
    _copy(
        shared.linear_fc1.weight,
        _tp_slice(shared_packed, 0, tp_rank, tp_size, f"{prefix}.shared.linear_fc1"),
        f"{prefix}.shared.linear_fc1",
    )
    shared_down = _require(source, f"{prefix}.fc2.weight")
    _copy(
        shared.linear_fc2.weight,
        _tp_slice(shared_down, 1, tp_rank, tp_size, f"{prefix}.shared.linear_fc2"),
        f"{prefix}.shared.linear_fc2",
    )


@torch.no_grad()
def load_xllm_mova_layer(layer, source: Mapping[str, Tensor], layer_idx: int) -> None:
    """Load one full xLLM layer into its local MCore PP/TP/EP shard."""

    if not layer.config.rotary_interleaved:
        raise ValueError("Native xLLM Q/K weights require rotary_interleaved=True")
    _copy(
        layer.input_layernorm.weight,
        _require(source, f"layers.{layer_idx}.norm.weight"),
        f"layers.{layer_idx}.norm.weight",
    )
    if layer_idx < layer.config.mova_num_dense_layers:
        _load_dense_attention(layer, source, layer_idx)
        _copy(
            layer.pre_mlp_layernorm.weight,
            _require(source, f"layers.{layer_idx}.nffn.norm.weight"),
            f"layers.{layer_idx}.nffn.norm.weight",
        )
        _load_dense_mlp(layer, source, layer_idx)
    else:
        _load_mova_attention(layer, source, layer_idx)
        _copy(
            layer.pre_mlp_layernorm.weight,
            _require(source, f"layers.{layer_idx}.moe.norm.weight"),
            f"layers.{layer_idx}.moe.norm.weight",
        )
        _load_sparse_mlp(layer, source, layer_idx)


@torch.no_grad()
def load_xllm_mova_global_weights(model, source: Mapping[str, Tensor]) -> None:
    """Load embedding, final norm, and output weights present on this PP rank."""

    tp_group = model.pg_collection.tp
    tp_rank, tp_size = get_pg_rank(tp_group), get_pg_size(tp_group)
    if model.pre_process:
        embedding = _require(source, "embed.weight")
        _copy(
            model.embedding.word_embeddings.weight,
            _tp_slice(embedding, 0, tp_rank, tp_size, "embed.weight"),
            "embed.weight",
        )
    if model.post_process:
        _copy(
            model.decoder.final_layernorm.weight,
            _require(source, "output.final_norm.weight"),
            "output.final_norm.weight",
        )
        output = _require(source, "output.output.weight")
        _copy(
            model.output_layer.weight,
            _tp_slice(output, 0, tp_rank, tp_size, "output.output.weight"),
            "output.output.weight",
        )


@torch.no_grad()
def load_xllm_mova_state_dict(model, source: Mapping[str, Tensor]) -> None:
    """Load every layer resident in one MCore GPT model chunk."""

    for layer in model.decoder.layers:
        load_xllm_mova_layer(layer, source, layer.layer_number - 1)
    load_xllm_mova_global_weights(model, source)
