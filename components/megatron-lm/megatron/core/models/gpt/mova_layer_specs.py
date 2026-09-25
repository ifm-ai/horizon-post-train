# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPT layer specifications for the xLLM MoVA architecture."""

from typing import Optional

import torch.nn.functional as F

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.mova import (
    GroupedGemmMoVAValueExperts,
    GroupRMSNorm,
    MoVASelfAttention,
    MoVASelfAttentionSubmodules,
    MoVATransformerConfig,
    MoVAValueProjection,
    MoVAValueProjectionSubmodules,
    SequentialMoVAValueExperts,
    SequentialMoVAValueExpertsSubmodules,
    SoftplusGatedSelfAttention,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)


def _get_dense_layer_spec(backend: BackendSpecProvider) -> ModuleSpec:
    """Build the dense-prefix layer without fusing away GroupRMSNorm."""

    attention = ModuleSpec(
        module=SoftplusGatedSelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=SelfAttentionSubmodules(
            linear_qkv=backend.column_parallel_linear(),
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            k_layernorm=IdentityOp,
        ),
    )
    mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=backend.column_parallel_linear(), linear_fc2=backend.row_parallel_linear()
        ),
    )
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=GroupRMSNorm,
            self_attention=attention,
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=GroupRMSNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def _get_mova_layer_spec(
    backend: BackendSpecProvider,
    value_backend: str,
    num_moe_experts: int,
    moe_grouped_gemm: bool,
    moe_use_legacy_grouped_gemm: bool,
) -> ModuleSpec:
    value_expert_module = (
        GroupedGemmMoVAValueExperts
        if value_backend == "grouped_gemm"
        else SequentialMoVAValueExperts
    )
    value_experts = ModuleSpec(
        module=value_expert_module,
        submodules=SequentialMoVAValueExpertsSubmodules(linear=backend.column_parallel_linear()),
    )
    value_projection = ModuleSpec(
        module=MoVAValueProjection, submodules=MoVAValueProjectionSubmodules(experts=value_experts)
    )
    attention = ModuleSpec(
        module=MoVASelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=MoVASelfAttentionSubmodules(
            linear_qkg=backend.column_parallel_linear(),
            value_projection=value_projection,
            core_attention=backend.core_attention(),
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=IdentityOp,
            k_layernorm=IdentityOp,
        ),
    )
    mlp = get_moe_module_spec_for_backend(
        backend=backend,
        num_experts=num_moe_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )
    if mlp.submodules.experts.module is TEGroupedMLP:
        # Native xLLM, HF, and SGLang weight the completed expert down
        # projection. Restrict that numerical contract to the production MoVA
        # TE grouped-expert path; other MoE specifications keep their default.
        mlp.submodules.experts.params["apply_router_probs_after_fc2"] = True
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=GroupRMSNorm,
            self_attention=attention,
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=GroupRMSNorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def get_mova_gpt_decoder_block_spec(
    config: MoVATransformerConfig,
    use_transformer_engine: bool = True,
    moe_grouped_gemm: bool = True,
    moe_use_legacy_grouped_gemm: bool = False,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
) -> TransformerBlockSubmodules:
    """Return the dense-prefix/MoVA block assigned to one pipeline stage."""

    if config.mova_num_value_experts <= 0:
        raise ValueError("MoVA block spec requires mova_num_value_experts > 0")
    if config.num_moe_experts is None:
        raise ValueError("MoVA sparse layers require feed-forward MoE experts")

    backend: BackendSpecProvider
    backend = TESpecProvider() if use_transformer_engine else LocalSpecProvider()
    dense_layer_spec = _get_dense_layer_spec(backend)
    mova_layer_spec = _get_mova_layer_spec(
        backend,
        value_backend=config.mova_value_backend,
        num_moe_experts=config.num_moe_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )
    layer_specs = [
        dense_layer_spec if layer_id < config.mova_num_dense_layers else mova_layer_spec
        for layer_id in range(config.num_layers)
    ]

    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    if config.pipeline_model_parallel_layout is not None:
        local_layer_specs = [
            layer_specs[layer_id]
            for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank
            )
        ]
    else:
        if pp_rank is None:
            # get_transformer_layer_offset resolves the active PP rank.
            offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        else:
            offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
        local_layer_specs = layer_specs[offset : offset + num_layers_to_build]

    return TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=GroupRMSNorm)


def get_k2mova_36b_config(**overrides) -> MoVATransformerConfig:
    """Exact architecture defaults for the xLLM 36B MoVA model family.

    Runtime choices such as parallel sizes, precision, recomputation, and the
    run-specific RoPE base remain explicit caller inputs.
    """

    defaults = dict(
        num_layers=48,
        hidden_size=2560,
        num_attention_heads=32,
        num_query_groups=8,
        kv_channels=128,
        ffn_hidden_size=6144,
        num_moe_experts=100,
        moe_ffn_hidden_size=768,
        moe_router_topk=8,
        moe_router_score_function="sigmoid",
        moe_router_topk_scaling_factor=2.5,
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=1.0e-3,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_shared_expert_intermediate_size=768,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        layernorm_epsilon=1.0e-6,
        attention_output_gate=True,
        moe_router_dtype="fp32",
        mova_num_value_experts=64,
        mova_router_topk=4,
        mova_router_score_function="sigmoid",
        mova_router_topk_scaling_factor=2.5,
        mova_router_enable_expert_bias=True,
        mova_router_bias_update_rate=1.0e-3,
        mova_router_load_balancing_type="none",
        mova_router_aux_loss_coeff=0.0,
        mova_num_dense_layers=3,
        mova_norm_num_groups=2,
        mova_attention_gate_function="softplus",
        mova_value_backend="grouped_gemm",
        xllm_router_compatibility=True,
        rotary_interleaved=True,
    )
    defaults.update(overrides)
    return MoVATransformerConfig(**defaults)
