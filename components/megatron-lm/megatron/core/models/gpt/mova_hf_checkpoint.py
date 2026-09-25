# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Strict, streaming reader for xLLM MoVA Hugging Face exports.

The Hugging Face bridge stores one complete model layer per PyTorch shard.  We
validate the full index before loading tensors, mmap one shard at a time, and
translate its public Hugging Face names into the native xLLM names consumed by
``mova_checkpoint.py``.  No Transformers dependency or remote code execution
is needed for checkpoint conversion.
"""

import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

HF_CONFIG_NAME = "config.json"
HF_INDEX_NAME = "pytorch_model.bin.index.json"
HF_COMPLETION_MARKER = "done.txt"


def _require(config: Mapping[str, Any], key: str) -> Any:
    try:
        return config[key]
    except KeyError as error:
        raise ValueError(f"Missing required xLLM Hugging Face config field: {key}") from error


def _require_positive_int(config: Mapping[str, Any], key: str) -> int:
    value = _require(config, key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"xLLM Hugging Face {key} must be a positive integer; got {value!r}")
    return value


def validate_hf_xllm_mova_config(config: Mapping[str, Any]) -> None:
    """Reject exports whose forward contract is not represented by this port."""

    exact = {
        "model_type": "xllm",
        "apply_attn_gate": True,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_gate_func": "softplus",
        "decoder_sparse_step": 1,
        "hidden_act": "silu",
        "moe_gate_bias": True,
        "norm_topk_prob": True,
        "query_key_norm": False,
        "router_score_func": "sigmoid",
        "sliding_window": None,
        "tie_word_embeddings": False,
        "use_sliding_window": False,
    }
    for key, expected in exact.items():
        actual = config.get(key)
        if actual != expected:
            raise ValueError(
                f"Unsupported xLLM Hugging Face {key}={actual!r}; expected {expected!r}"
            )

    positive_fields = (
        "hidden_size",
        "head_dim",
        "intermediate_size",
        "layernorm_num_groups",
        "max_position_embeddings",
        "moe_intermediate_size",
        "num_attention_heads",
        "num_experts",
        "num_experts_per_tok",
        "num_hidden_layers",
        "num_key_value_heads",
        "num_shared_experts",
        "num_values",
        "num_values_per_tok",
        "vocab_size",
    )
    values = {key: _require_positive_int(config, key) for key in positive_fields}
    num_dense_layers = _require(config, "num_dense_layers")
    if (
        isinstance(num_dense_layers, bool)
        or not isinstance(num_dense_layers, int)
        or not 0 <= num_dense_layers < values["num_hidden_layers"]
    ):
        raise ValueError("xLLM Hugging Face num_dense_layers is invalid")
    if config.get("mlp_only_layers") != list(range(num_dense_layers)):
        raise ValueError("Only a contiguous dense-prefix layer layout is supported")
    if values["num_shared_experts"] != 1:
        raise ValueError("The MoVA mapping requires exactly one shared FFN expert")
    if values["num_attention_heads"] % values["num_key_value_heads"]:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if values["head_dim"] % 2:
        raise ValueError("head_dim must be even for xLLM complex RoPE")
    if values["hidden_size"] % values["layernorm_num_groups"]:
        raise ValueError("hidden_size must be divisible by layernorm_num_groups")
    if values["num_experts_per_tok"] > values["num_experts"]:
        raise ValueError("num_experts_per_tok exceeds num_experts")
    if values["num_values_per_tok"] > values["num_values"]:
        raise ValueError("num_values_per_tok exceeds num_values")
    if config.get("rope_head_dim") != values["head_dim"]:
        raise ValueError("Only full-head RoPE is supported")
    if config.get("rope_scaling") is not None:
        raise ValueError("RoPE scaling is not represented by the current MoVA port")
    if (
        isinstance(config.get("rope_theta"), bool)
        or not isinstance(config.get("rope_theta"), (int, float))
        or not math.isfinite(config["rope_theta"])
        or config["rope_theta"] <= 0
    ):
        raise ValueError("rope_theta must be positive")
    if (
        isinstance(config.get("rms_norm_eps"), bool)
        or not isinstance(config.get("rms_norm_eps"), (int, float))
        or not math.isfinite(config["rms_norm_eps"])
        or config["rms_norm_eps"] <= 0
    ):
        raise ValueError("rms_norm_eps must be positive")
    if (
        isinstance(config.get("router_scaling_factor"), bool)
        or not isinstance(config.get("router_scaling_factor"), (int, float))
        or not math.isfinite(config["router_scaling_factor"])
        or config["router_scaling_factor"] <= 0
    ):
        raise ValueError("router_scaling_factor must be positive")
    # Schema-check the HF bridge coefficient, but do not use it as native
    # training metadata; current xLLM exports populate it from a default.
    if (
        isinstance(config.get("router_aux_loss_coef"), bool)
        or not isinstance(config.get("router_aux_loss_coef"), (int, float))
        or not math.isfinite(config["router_aux_loss_coef"])
        or config["router_aux_loss_coef"] < 0
    ):
        raise ValueError("router_aux_loss_coef must be non-negative")


def hf_xllm_config_to_source_config(
    config: Mapping[str, Any],
    *,
    router_gemm_partitions: int,
    router_bias_update_rate: float,
    router_load_balancing_type: str | None,
    router_aux_loss_coeff: float,
) -> dict[str, Any]:
    """Translate an HF export config into the normalized native-xLLM schema.

    The current xLLM exporter does not preserve native router-training
    settings: its ``router_aux_loss_coef`` is an HF bridge default. Require
    those settings from the source run, as we already do for router GEMM
    partitions and the loss-free bias update rate.
    """

    validate_hf_xllm_mova_config(config)
    hidden_size = config["hidden_size"]
    if (
        isinstance(router_gemm_partitions, bool)
        or not isinstance(router_gemm_partitions, int)
        or router_gemm_partitions <= 0
        or hidden_size % router_gemm_partitions
    ):
        raise ValueError("router_gemm_partitions must be positive and divide hidden_size")
    if (
        isinstance(router_bias_update_rate, bool)
        or not isinstance(router_bias_update_rate, (int, float))
        or not math.isfinite(router_bias_update_rate)
        or router_bias_update_rate <= 0
    ):
        raise ValueError("router_bias_update_rate must be positive")
    if router_load_balancing_type not in (None, "dot"):
        raise ValueError("router_load_balancing_type must be none or dot")
    if (
        isinstance(router_aux_loss_coeff, bool)
        or not isinstance(router_aux_loss_coeff, (int, float))
        or not math.isfinite(router_aux_loss_coeff)
        or router_aux_loss_coeff < 0
    ):
        raise ValueError("router_aux_loss_coeff must be non-negative")
    if router_load_balancing_type is None and router_aux_loss_coeff != 0:
        raise ValueError("router_aux_loss_coeff must be zero when load balancing is disabled")
    if router_load_balancing_type == "dot" and router_aux_loss_coeff == 0:
        raise ValueError("router_aux_loss_coeff must be positive for dot load balancing")

    return {
        "seq_len": config["max_position_embeddings"],
        "model_parallel_size": router_gemm_partitions,
        "moe_router_load_balancing_type": router_load_balancing_type,
        "moe_aux_loss_coeff": float(router_aux_loss_coeff),
        "model": {
            "arch": "transformer",
            "num_layers": config["num_hidden_layers"],
            "model_dim": hidden_size,
            "num_heads": config["num_attention_heads"],
            "num_kv_heads": config["num_key_value_heads"],
            "head_dim": config["head_dim"],
            "rope_head_dim": config["rope_head_dim"],
            "ffn_hidden_dim": config["intermediate_size"],
            "num_experts": config["num_experts"],
            "num_activated_experts": config["num_experts_per_tok"],
            "num_shared_experts": config["num_shared_experts"],
            "expert_inter_dim": config["moe_intermediate_size"],
            "num_values": config["num_values"],
            "num_activated_values": config["num_values_per_tok"],
            "num_dense_layers": config["num_dense_layers"],
            "moe_router_score_func": config["router_score_func"],
            "moe_router_scaling_factor": config["router_scaling_factor"],
            "moe_router_bias": config["moe_gate_bias"],
            "moe_router_bias_update_rate": router_bias_update_rate,
            "layernorm_num_groups": config["layernorm_num_groups"],
            "norm_eps": config["rms_norm_eps"],
            "rope_base": config["rope_theta"],
            "attn_gate_func": config["attn_gate_func"],
            "vocab_size": config["vocab_size"],
            "output_size": config["vocab_size"],
            "apply_rmsnorm": True,
            "apply_attn_gate": config["apply_attn_gate"],
            "qknorm": config["query_key_norm"],
            "swiglu": True,
            "two_hop_residual": False,
            "scale_emb": False,
            "rescale_nffn": False,
            "dropout": 0.0,
            "hidden_dropout": 0.0,
            "attention_dropout": config["attention_dropout"],
        },
    }


def _dense_key_map(layer_idx: int) -> dict[str, str]:
    hf = f"model.layers.{layer_idx}"
    native = f"layers.{layer_idx}"
    return {
        f"{hf}.self_attn.q_proj.weight": f"{native}.attention.wq.weight",
        f"{hf}.self_attn.k_proj.weight": f"{native}.attention.wk.weight",
        f"{hf}.self_attn.v_proj.weight": f"{native}.attention.wv.weight",
        f"{hf}.self_attn.attn_gate_proj.weight": f"{native}.attention.wr.weight",
        f"{hf}.self_attn.o_proj.weight": f"{native}.attention.wo.weight",
        f"{hf}.input_layernorm.weight": f"{native}.norm.weight",
        f"{hf}.post_attention_layernorm.weight": f"{native}.nffn.norm.weight",
        f"{hf}.mlp.gate_proj.weight": f"{native}.nffn.fc1.weight",
        f"{hf}.mlp.down_proj.weight": f"{native}.nffn.fc2.weight",
        f"{hf}.mlp.up_proj.weight": f"{native}.nffn.fc3.weight",
    }


def _sparse_direct_key_map(layer_idx: int) -> dict[str, str]:
    hf = f"model.layers.{layer_idx}"
    native = f"layers.{layer_idx}"
    return {
        f"{hf}.self_attn.q_proj.weight": f"{native}.mova.wq.weight",
        f"{hf}.self_attn.k_proj.weight": f"{native}.mova.wk.weight",
        f"{hf}.self_attn.attn_gate_proj.weight": f"{native}.mova.wr.weight",
        f"{hf}.self_attn.o_proj.weight": f"{native}.mova.wo.weight",
        f"{hf}.self_attn.v_router.weight": f"{native}.mova.router.weight",
        f"{hf}.self_attn.v_router.bias": f"{native}.mova.router.bias",
        f"{hf}.input_layernorm.weight": f"{native}.norm.weight",
        f"{hf}.post_attention_layernorm.weight": f"{native}.moe.norm.weight",
        f"{hf}.mlp.gate.weight": f"{native}.moe.router.weight",
        f"{hf}.mlp.gate.bias": f"{native}.moe.router.bias",
        f"{hf}.mlp.shared_experts.gate_proj.weight": f"{native}.moe.fc1.weight",
        f"{hf}.mlp.shared_experts.down_proj.weight": f"{native}.moe.fc2.weight",
        f"{hf}.mlp.shared_experts.up_proj.weight": f"{native}.moe.fc3.weight",
    }


_GLOBAL_KEY_MAP = {
    "model.embed_tokens.weight": "embed.weight",
    "model.norm.weight": "output.final_norm.weight",
    "lm_head.weight": "output.output.weight",
}


def expected_hf_xllm_mova_layer_keys(config: Mapping[str, Any], layer_idx: int) -> set[str]:
    num_layers = config["num_hidden_layers"]
    if not 0 <= layer_idx < num_layers:
        raise IndexError(f"Layer {layer_idx} is outside [0, {num_layers})")
    if layer_idx < config["num_dense_layers"]:
        return set(_dense_key_map(layer_idx))

    prefix = f"model.layers.{layer_idx}"
    keys = set(_sparse_direct_key_map(layer_idx))
    keys.update(
        f"{prefix}.self_attn.v_experts.{expert_idx}.weight"
        for expert_idx in range(config["num_values"])
    )
    for expert_idx in range(config["num_experts"]):
        keys.update(
            {
                f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight",
                f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight",
                f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight",
            }
        )
    return keys


def expected_hf_xllm_mova_keys(config: Mapping[str, Any]) -> set[str]:
    keys = set(_GLOBAL_KEY_MAP)
    for layer_idx in range(config["num_hidden_layers"]):
        keys.update(expected_hf_xllm_mova_layer_keys(config, layer_idx))
    return keys


def _require_tensors(source: Mapping[str, Tensor], keys: set[str]) -> None:
    missing = sorted(keys.difference(source))
    if missing:
        raise KeyError(f"Missing required Hugging Face tensors: {missing[:4]}")
    non_tensors = sorted(key for key in keys if not isinstance(source[key], Tensor))
    if non_tensors:
        raise TypeError(f"Hugging Face checkpoint entries are not tensors: {non_tensors[:4]}")


def _unpermute_hf_qk(
    weight: Tensor, *, num_heads: int, head_dim: int, hidden_size: int, key: str
) -> Tensor:
    """Restore adjacent complex-pair channels from the xLLM HF export layout."""

    expected_shape = (num_heads * head_dim, hidden_size)
    if tuple(weight.shape) != expected_shape:
        raise ValueError(
            f"Invalid Hugging Face {key} shape: got {tuple(weight.shape)}, "
            f"expected {expected_shape}"
        )
    # The xLLM->HF bridge changes [head, half, pair, hidden] into
    # [head, pair, half, hidden]. Undo it before using interleaved RoPE in MCore.
    return (
        weight.reshape(num_heads, 2, head_dim // 2, hidden_size)
        .transpose(1, 2)
        .reshape(num_heads * head_dim, hidden_size)
        .contiguous()
    )


def normalize_hf_xllm_mova_layer_state(
    source: Mapping[str, Tensor], config: Mapping[str, Any], layer_idx: int
) -> dict[str, Tensor]:
    """Map one HF layer shard to the native xLLM checkpoint naming contract."""

    expected = expected_hf_xllm_mova_layer_keys(config, layer_idx)
    _require_tensors(source, expected)
    hf_prefix = f"model.layers.{layer_idx}.self_attn"
    native_prefix = f"layers.{layer_idx}"
    if layer_idx < config["num_dense_layers"]:
        normalized = {native: source[hf] for hf, native in _dense_key_map(layer_idx).items()}
        attention_prefix = f"{native_prefix}.attention"
    else:
        normalized = {
            native: source[hf] for hf, native in _sparse_direct_key_map(layer_idx).items()
        }
        attention_prefix = f"{native_prefix}.mova"

    for projection, num_heads in (
        ("q", config["num_attention_heads"]),
        ("k", config["num_key_value_heads"]),
    ):
        hf_key = f"{hf_prefix}.{projection}_proj.weight"
        normalized[f"{attention_prefix}.w{projection}.weight"] = _unpermute_hf_qk(
            source[hf_key],
            num_heads=num_heads,
            head_dim=config["head_dim"],
            hidden_size=config["hidden_size"],
            key=hf_key,
        )

    if layer_idx < config["num_dense_layers"]:
        return normalized

    hf = f"model.layers.{layer_idx}"
    native = f"layers.{layer_idx}"
    normalized[f"{native}.mova.wv.weight"] = torch.stack(
        [
            source[f"{hf}.self_attn.v_experts.{expert_idx}.weight"]
            for expert_idx in range(config["num_values"])
        ]
    )
    normalized[f"{native}.moe.experts.weight1"] = torch.stack(
        [
            source[f"{hf}.mlp.experts.{expert_idx}.gate_proj.weight"]
            for expert_idx in range(config["num_experts"])
        ]
    )
    normalized[f"{native}.moe.experts.weight2"] = torch.stack(
        [
            source[f"{hf}.mlp.experts.{expert_idx}.down_proj.weight"]
            for expert_idx in range(config["num_experts"])
        ]
    )
    normalized[f"{native}.moe.experts.weight3"] = torch.stack(
        [
            source[f"{hf}.mlp.experts.{expert_idx}.up_proj.weight"]
            for expert_idx in range(config["num_experts"])
        ]
    )
    return normalized


def normalize_hf_xllm_mova_global_state(source: Mapping[str, Tensor]) -> dict[str, Tensor]:
    _require_tensors(source, set(_GLOBAL_KEY_MAP))
    return {native: source[hf] for hf, native in _GLOBAL_KEY_MAP.items()}


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {description} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def _describe_key_difference(expected: set[str], actual: set[str]) -> str:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    return f"missing={missing[:4]}, unexpected={unexpected[:4]}"


class HFXllmMoVACheckpoint:
    """Validated metadata plus bounded, mmap-backed layer reads."""

    def __init__(self, checkpoint_dir: Path | str, *, require_done: bool = True) -> None:
        self.path = Path(checkpoint_dir)
        if not self.path.is_dir():
            raise FileNotFoundError(f"HF checkpoint directory not found: {self.path}")
        if require_done and not (self.path / HF_COMPLETION_MARKER).is_file():
            raise FileNotFoundError(
                f"HF checkpoint completion marker is missing: {self.path / HF_COMPLETION_MARKER}"
            )

        self.config = _read_json(self.path / HF_CONFIG_NAME, "xLLM Hugging Face config")
        validate_hf_xllm_mova_config(self.config)
        index = _read_json(self.path / HF_INDEX_NAME, "Hugging Face weight index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Hugging Face index has no non-empty weight_map")
        if not all(
            isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()
        ):
            raise ValueError("Hugging Face weight_map must map strings to shard filenames")
        self.weight_map: dict[str, str] = weight_map

        expected = expected_hf_xllm_mova_keys(self.config)
        actual = set(self.weight_map)
        if actual != expected:
            raise ValueError(
                "Hugging Face index does not match the supported MoVA architecture: "
                + _describe_key_difference(expected, actual)
            )

        self._shard_keys: dict[str, set[str]] = {}
        for key, shard_name in self.weight_map.items():
            if Path(shard_name).name != shard_name or not shard_name.endswith(".bin"):
                raise ValueError(f"Unsafe or unsupported checkpoint shard name: {shard_name!r}")
            self._shard_keys.setdefault(shard_name, set()).add(key)
        for shard_name in self._shard_keys:
            if not (self.path / shard_name).is_file():
                raise FileNotFoundError(f"Missing checkpoint shard: {self.path / shard_name}")

        self._layer_shards = []
        for layer_idx in range(self.config["num_hidden_layers"]):
            layer_keys = expected_hf_xllm_mova_layer_keys(self.config, layer_idx)
            shard_names = {self.weight_map[key] for key in layer_keys}
            if len(shard_names) != 1:
                raise ValueError(f"Layer {layer_idx} spans multiple Hugging Face shards")
            self._layer_shards.append(next(iter(shard_names)))
        global_shards = {self.weight_map[key] for key in _GLOBAL_KEY_MAP}
        if len(global_shards) != 1:
            raise ValueError("Embedding, final norm, and output weights span multiple shards")
        self._global_shard = next(iter(global_shards))

    @property
    def num_layers(self) -> int:
        return self.config["num_hidden_layers"]

    def source_config(
        self,
        *,
        router_gemm_partitions: int,
        router_bias_update_rate: float,
        router_load_balancing_type: str | None,
        router_aux_loss_coeff: float,
    ) -> dict[str, Any]:
        return hf_xllm_config_to_source_config(
            self.config,
            router_gemm_partitions=router_gemm_partitions,
            router_bias_update_rate=router_bias_update_rate,
            router_load_balancing_type=router_load_balancing_type,
            router_aux_loss_coeff=router_aux_loss_coeff,
        )

    def layer_shard(self, layer_idx: int) -> str:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"Layer {layer_idx} is outside [0, {self.num_layers})")
        return self._layer_shards[layer_idx]

    @property
    def global_shard(self) -> str:
        return self._global_shard

    def _load_shard(self, shard_name: str) -> Mapping[str, Tensor]:
        shard_path = self.path / shard_name
        source = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(source, Mapping):
            raise TypeError(f"Checkpoint shard is not a state dict: {shard_path}")
        expected = self._shard_keys[shard_name]
        actual = set(source)
        if actual != expected:
            raise ValueError(
                f"Shard {shard_name} disagrees with its index: "
                + _describe_key_difference(expected, actual)
            )
        return source

    def load_layer(self, layer_idx: int, *, include_globals: bool = False) -> dict[str, Tensor]:
        shard_name = self.layer_shard(layer_idx)
        if include_globals and shard_name != self.global_shard:
            raise ValueError("Requested globals are not colocated with this layer shard")
        source = self._load_shard(shard_name)
        normalized = normalize_hf_xllm_mova_layer_state(source, self.config, layer_idx)
        if include_globals:
            normalized.update(normalize_hf_xllm_mova_global_state(source))
        return normalized

    def load_globals(self) -> dict[str, Tensor]:
        return normalize_hf_xllm_mova_global_state(self._load_shard(self.global_shard))
