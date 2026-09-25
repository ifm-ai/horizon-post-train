from mbridge.core import register_model
from mbridge.models import Qwen2Bridge, Qwen2MoEBridge


@register_model(["xllm", "k2_horizon"])
class XllmBridge(Qwen2MoEBridge):
    """Bridge implementation for xLLM and K2 Horizon checkpoints."""

    _MLP_MAPPING = {
        **Qwen2MoEBridge._MLP_MAPPING,
        **Qwen2Bridge._MLP_MAPPING,
        "mlp.router.expert_bias": ["model.layers.{layer_number}.mlp.gate.bias"],
        "shared_experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.shared_experts.gate_proj.weight",
            "model.layers.{layer_number}.mlp.shared_experts.up_proj.weight",
        ],
        "shared_experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.shared_experts.down_proj.weight",
        ],
    }

    def _has_moe(self):
        return (getattr(self.hf_config, "num_experts", 0) or 0) > 0

    def _build_config(self):
        head_dim = getattr(
            self.hf_config,
            "head_dim",
            self.hf_config.hidden_size // self.hf_config.num_attention_heads,
        )
        rope_head_dim = getattr(self.hf_config, "rope_head_dim", head_dim)

        config_kwargs = dict(
            use_cpu_initialization=False,
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            qk_layernorm=False,
            add_qkv_bias=False,
            add_bias_linear=False,
            rotary_percent=rope_head_dim / head_dim,
            xllm_partial_rope_layout=rope_head_dim * 2 == head_dim,
        )

        if self._has_moe():
            config_kwargs.update(
                moe_ffn_hidden_size=self.hf_config.moe_intermediate_size,
                moe_router_bias_update_rate=0,
                moe_router_topk=self.hf_config.num_experts_per_tok,
                num_moe_experts=self.hf_config.num_experts,
                moe_router_load_balancing_type="none",
                moe_grouped_gemm=True,
                moe_router_score_function="sigmoid",
                moe_router_enable_expert_bias=True,
                moe_router_pre_softmax=True,
            )

        return self._build_base_config(**config_kwargs)

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name, "extra_state should not be loaded"
        direct_name_mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }
        if mcore_weights_name in direct_name_mapping:
            return [direct_name_mapping[mcore_weights_name]]

        layer_prefix = "decoder.layers."
        if mcore_weights_name.startswith(layer_prefix):
            layer_idx, _, rest = mcore_weights_name[len(layer_prefix) :].partition(".")
            if rest == "input_layernorm.weight":
                return [f"model.layers.{layer_idx}.input_layernorm.weight"]
            if rest == "pre_mlp_layernorm.weight":
                return [f"model.layers.{layer_idx}.post_attention_layernorm.weight"]

        if "self_attention" in mcore_weights_name:
            return self._weight_name_mapping_attention(mcore_weights_name)
        if "mlp" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")
