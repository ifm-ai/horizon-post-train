import re

import torch


def _is_mova(args) -> bool:
    return getattr(args, "mova_num_value_experts", 0) > 0


def _attention_geometry(args) -> tuple[int, int, int, int]:
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    num_query_groups = args.num_query_groups
    kv_channels = getattr(args, "kv_channels", None)
    head_dim = kv_channels if kv_channels is not None else hidden_size // num_attention_heads

    if num_attention_heads % num_query_groups:
        raise ValueError(
            f"num_attention_heads={num_attention_heads} must be divisible by " f"num_query_groups={num_query_groups}"
        )
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"xLLM MoVA requires an even positive head dimension, got {head_dim}")
    return hidden_size, num_attention_heads, num_query_groups, head_dim


def _permute_qk_to_hf(
    weight: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    hidden_size: int,
    name: str,
) -> torch.Tensor:
    """Convert MCore's adjacent-complex-pair Q/K rows to xLLM HF layout."""

    expected_shape = (num_heads * head_dim, hidden_size)
    if tuple(weight.shape) != expected_shape:
        raise ValueError(f"Invalid {name} shape: got {tuple(weight.shape)}, expected {expected_shape}")
    return (
        weight.reshape(num_heads, head_dim // 2, 2, hidden_size).transpose(1, 2).reshape(expected_shape).contiguous()
    )


def _unpack_grouped_attention_projection(
    args,
    name: str,
    param: torch.Tensor,
    *,
    include_value: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Unpack MCore's per-GQA-group Q/G/K[/V] projection."""

    hidden_size, num_attention_heads, num_query_groups, head_dim = _attention_geometry(args)
    query_heads_per_group = num_attention_heads // num_query_groups
    segment_heads = [query_heads_per_group, query_heads_per_group, 1]
    if include_value:
        segment_heads.append(1)

    expected_rows = num_query_groups * sum(segment_heads) * head_dim
    if tuple(param.shape) != (expected_rows, hidden_size):
        raise ValueError(
            f"Invalid {name} shape: got {tuple(param.shape)}, expected " f"{(expected_rows, hidden_size)}"
        )

    packed = param.reshape(num_query_groups, sum(segment_heads), head_dim, hidden_size)
    chunks = torch.split(packed, segment_heads, dim=1)
    query = chunks[0].reshape(num_attention_heads * head_dim, hidden_size)
    gate = chunks[1].reshape(num_attention_heads * head_dim, hidden_size)
    key = chunks[2].reshape(num_query_groups * head_dim, hidden_size)
    value = chunks[3].reshape(num_query_groups * head_dim, hidden_size) if include_value else None

    query = _permute_qk_to_hf(
        query,
        num_heads=num_attention_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        name=f"{name}.query",
    )
    key = _permute_qk_to_hf(
        key,
        num_heads=num_query_groups,
        head_dim=head_dim,
        hidden_size=hidden_size,
        name=f"{name}.key",
    )
    return query, gate, key, value


def _convert_grouped_value_experts(
    args, layer_idx: str, name: str, param: torch.Tensor
) -> list[tuple[str, torch.Tensor]]:
    """Convert gathered MCore Wv [expert, hidden, value] to HF [value, hidden]."""

    if param.ndim != 3:
        raise ValueError(
            f"Invalid {name} shape: got {tuple(param.shape)}, expected "
            "[num_value_experts, hidden_size, value_width]"
        )
    expected_experts = getattr(args, "mova_num_value_experts", 0)
    if expected_experts and param.shape[0] != expected_experts:
        raise ValueError(f"Invalid {name} expert count: got {param.shape[0]}, expected {expected_experts}")
    if param.shape[1] != args.hidden_size:
        raise ValueError(
            f"Invalid {name} hidden dimension: got {param.shape[1]}, expected {args.hidden_size}. "
            "The grouped MoVA weight must be gathered across regular attention TP before conversion."
        )

    return [
        (
            f"model.layers.{layer_idx}.self_attn.v_experts.{expert_idx}.weight",
            expert_weight.transpose(0, 1).contiguous(),
        )
        for expert_idx, expert_weight in enumerate(param.unbind(dim=0))
    ]


def convert_xllm_to_hf(args, name, param):
    """Convert Megatron parameter names/tensors to HuggingFace xLLM format.

    MoVA uses MCore's interleaved Q/G/K[/V] projections and a grouped value
    weight stored as ``[expert, hidden / TP, value_width]``. The caller first
    gathers those tensors over regular attention TP; this function then emits
    canonical xLLM HF names consumed by both SGLang broadcast and P2P loading.
    """

    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    hidden_size, num_attention_heads, num_query_groups, head_dim = _attention_geometry(args)
    query_heads_per_group = num_attention_heads // num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
            if rest == "linear_fc2":
                return [(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param)]
            raise ValueError(f"Unknown expert parameter name: {name}")

        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight", up_weight),
                ]
            if rest == "linear_fc2.weight":
                return [(f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight", param)]
            raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]

        if rest == "self_attention.linear_qkv.weight":
            if _is_mova(args):
                query, gate, key, value = _unpack_grouped_attention_projection(args, name, param, include_value=True)
                assert value is not None
                return [
                    (f"model.layers.{layer_idx}.self_attn.q_proj.weight", query),
                    (f"model.layers.{layer_idx}.self_attn.attn_gate_proj.weight", gate),
                    (f"model.layers.{layer_idx}.self_attn.k_proj.weight", key),
                    (f"model.layers.{layer_idx}.self_attn.v_proj.weight", value),
                ]

            # Preserve the legacy xLLM Q/K/V contract when MoVA is disabled.
            packed = param.view(num_query_groups, -1, head_dim, hidden_size)
            query, key, value = torch.split(packed, [query_heads_per_group, 1, 1], dim=1)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", query.reshape(-1, hidden_size)),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", key.reshape(-1, hidden_size)),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", value.reshape(-1, hidden_size)),
            ]

        if rest == "self_attention.linear_qkg.weight":
            if not _is_mova(args):
                raise ValueError(f"Found MoVA Q/K/gate projection while MoVA is disabled: {name}")
            query, gate, key, value = _unpack_grouped_attention_projection(args, name, param, include_value=False)
            assert value is None
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", query),
                (f"model.layers.{layer_idx}.self_attn.attn_gate_proj.weight", gate),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", key),
            ]

        if rest == "self_attention.value_projection.experts.weight":
            if not _is_mova(args):
                raise ValueError(f"Found grouped MoVA value experts while MoVA is disabled: {name}")
            return _convert_grouped_value_experts(args, layer_idx, name, param)

        sequential_value_expert_pattern = r"self_attention\.value_projection\.experts\.experts\.(\d+)\.weight"
        match = re.match(sequential_value_expert_pattern, rest)
        if match:
            expert_idx = match.group(1)
            return [(f"model.layers.{layer_idx}.self_attn.v_experts.{expert_idx}.weight", param)]

        if rest == "self_attention.value_projection.router.weight":
            return [(f"model.layers.{layer_idx}.self_attn.v_router.weight", param)]
        if rest == "self_attention.value_projection.router.expert_bias":
            return [(f"model.layers.{layer_idx}.self_attn.v_router.bias", param)]

        if rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        if rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]

        if rest in (
            "self_attention.linear_qkv.layer_norm_weight",
            "self_attention.linear_qkg.layer_norm_weight",
            "input_layernorm.weight",
        ):
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        if rest in ("mlp.linear_fc1.layer_norm_weight", "pre_mlp_layernorm.weight"):
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        if rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        if rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.bias", param)]

    raise ValueError(f"Unknown parameter name: {name}")
