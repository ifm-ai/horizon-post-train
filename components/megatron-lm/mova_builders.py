# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model builder for xLLM-compatible Mixture-of-Value Attention GPTs."""

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.mova_layer_specs import get_mova_gpt_decoder_block_spec
from megatron.core.transformer.mova import MoVATransformerConfig
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args


def mova_builder(args, pre_process, post_process, vp_stage=None, config=None):
    """Build a GPT model containing a dense prefix followed by MoVA-MoE layers."""

    if args.use_legacy_models:
        raise ValueError("MoVA is supported only by Megatron Core models")
    if args.yaml_cfg is not None:
        raise ValueError("MoVA's initial integration supports CLI configuration only")
    if args.spec is not None:
        raise ValueError("MoVA owns its heterogeneous layer spec; --spec is not supported")
    if args.mtp_num_layers is not None:
        raise ValueError("MoVA does not currently support MTP layers")

    print_rank_0("building MoVA GPT model ...")
    if config is None:
        config = core_transformer_config_from_args(args, MoVATransformerConfig)

    transformer_layer_spec = get_mova_gpt_decoder_block_spec(
        config,
        use_transformer_engine=args.transformer_impl == "transformer_engine",
        moe_grouped_gemm=args.moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
        vp_stage=vp_stage,
    )
    return GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        vp_stage=vp_stage,
    )
