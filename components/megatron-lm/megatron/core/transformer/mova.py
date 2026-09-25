# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mixture-of-Value Attention (MoVA) components.

MoVA keeps standard causal grouped-query attention, but replaces the single
value projection with a token-routed mixture of value projections.  This file
contains only the model primitive; GPT layer composition lives in
``megatron.core.models.gpt.mova_layer_specs``.
"""

import copy
import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.mappings import (
    all_to_all_sp2hp,
    gather_from_sequence_parallel_region,
    reduce_scatter_last_dim_to_tensor_parallel_region,
)
from megatron.core.transformer.attention import Attention, SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe import grouped_gemm_util as grouped_gemm
from megatron.core.transformer.moe.moe_utils import permute, unpermute
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
)
from megatron.core.utils import divide, get_pg_rank, get_pg_size


@dataclass
class MoVATransformerConfig(TransformerConfig):
    """Transformer configuration fields owned by MoVA.

    MoVA routing is intentionally separate from feed-forward MoE routing: the
    production model uses 64 top-4 value experts and 100 top-8 MLP experts in
    the same layer.
    """

    mova_num_value_experts: int = 0
    """Number of routed value projections. Zero disables MoVA."""

    mova_router_topk: int = 1
    """Number of active value projections per token."""

    mova_router_score_function: str = "sigmoid"
    """Score function used by the value router."""

    mova_router_topk_scaling_factor: Optional[float] = 1.0
    """Scale applied after normalizing selected value-router scores."""

    mova_router_enable_expert_bias: bool = False
    """Use the loss-free selection bias for value routing."""

    mova_router_bias_update_rate: float = 1.0e-3
    """Update rate for the value router's loss-free selection bias."""

    mova_router_aux_loss_coeff: float = 0.0
    """Auxiliary load-balancing coefficient for value routing."""

    mova_router_load_balancing_type: str = "none"
    """Load-balancing objective for value routing."""

    mova_num_dense_layers: int = 0
    """Number of dense-prefix layers before MoVA layers."""

    mova_norm_num_groups: int = 1
    """Number of independent feature groups in zero-centered RMSNorm."""

    mova_attention_gate_function: str = "softplus"
    """Post-attention gate. The reference model uses softplus(beta=log(2))."""

    mova_value_backend: str = "sequential"
    """Value expert backend: auditable sequential or fused grouped GEMM."""

    mova_use_torch_rms_norm: bool = False
    """Use PyTorch's native RMSNorm primitive for grouped normalization."""

    xllm_router_compatibility: bool = False
    """Match xLLM's BF16 router GEMM followed by FP32 scoring and top-k."""

    xllm_router_gemm_partitions: int = 1
    """Reproduce xLLM's hidden-sharded router GEMM rounding without communication."""

    def __post_init__(self):
        super().__post_init__()
        if self.mova_num_value_experts < 0:
            raise ValueError("mova_num_value_experts must be non-negative")
        if self.mova_norm_num_groups <= 0:
            raise ValueError("mova_norm_num_groups must be positive")
        if self.mova_router_topk_scaling_factor is not None and (
            not math.isfinite(self.mova_router_topk_scaling_factor)
            or self.mova_router_topk_scaling_factor <= 0
        ):
            raise ValueError("mova_router_topk_scaling_factor must be positive")
        if (
            not math.isfinite(self.mova_router_bias_update_rate)
            or self.mova_router_bias_update_rate <= 0
        ):
            raise ValueError("mova_router_bias_update_rate must be positive")
        if (
            not math.isfinite(self.mova_router_aux_loss_coeff)
            or self.mova_router_aux_loss_coeff < 0
        ):
            raise ValueError("mova_router_aux_loss_coeff must be non-negative")
        if self.mova_num_value_experts:
            if not 0 < self.mova_router_topk <= self.mova_num_value_experts:
                raise ValueError("mova_router_topk must be in [1, mova_num_value_experts]")
            if self.mova_router_score_function not in ("sigmoid", "softmax"):
                raise ValueError("MoVA supports sigmoid or softmax routing")
            if self.mova_router_load_balancing_type not in ("none", "aux_loss"):
                raise ValueError("MoVA supports none or aux_loss load balancing")
            if self.mova_attention_gate_function not in ("softplus", "silu"):
                raise ValueError("MoVA supports softplus or silu attention gates")
            if self.mova_value_backend not in ("sequential", "grouped_gemm"):
                raise ValueError("MoVA value backend must be sequential or grouped_gemm")
        if not 0 <= self.mova_num_dense_layers <= self.num_layers:
            raise ValueError("mova_num_dense_layers must be in [0, num_layers]")
        if self.mova_num_value_experts and self.mova_num_dense_layers == self.num_layers:
            raise ValueError("MoVA requires at least one sparse value-expert layer")
        if self.xllm_router_gemm_partitions < 1:
            raise ValueError("xllm_router_gemm_partitions must be positive")
        if self.xllm_router_gemm_partitions != 1 and not self.xllm_router_compatibility:
            raise ValueError("xllm_router_gemm_partitions requires xllm_router_compatibility")
        if self.hidden_size % self.xllm_router_gemm_partitions:
            raise ValueError("hidden_size must be divisible by xllm_router_gemm_partitions")
        if self.hidden_size % self.mova_norm_num_groups:
            raise ValueError("hidden_size must be divisible by mova_norm_num_groups")
        if self.mova_num_value_experts and 0 < self.mova_num_dense_layers < self.num_layers:
            # Dense-prefix and MoVA layers have different parameter keys and
            # shapes, so distributed checkpoints must key layers explicitly.
            self.hetereogenous_dist_checkpoint = True


class GroupRMSNorm(MegatronModule):
    """RMSNorm over independent, contiguous feature groups.

    The scale is stored around zero and applied as ``1 + weight``, matching
    the xLLM checkpoint representation.  The eager implementation is the
    correctness reference; replacing its internals with a fused kernel does
    not change the state-dict contract.
    """

    def __init__(
        self, config: MoVATransformerConfig, hidden_size: int, eps: float = 1.0e-5, **_: object
    ) -> None:
        super().__init__(config=config)
        if hidden_size % config.mova_norm_num_groups:
            raise ValueError("hidden_size must be divisible by mova_norm_num_groups")
        self.hidden_size = hidden_size
        self.num_groups = config.mova_norm_num_groups
        self.group_size = hidden_size // self.num_groups
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, dtype=config.params_dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the zero-centered scale, including after meta materialization."""
        if self.config.perform_initialization and not self.weight.is_meta:
            with torch.no_grad():
                self.weight.zero_()
        setattr(self.weight, "sequence_parallel", self.config.sequence_parallel)

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        grouped = hidden_states.float().reshape(*hidden_states.shape[:-1], self.num_groups, -1)
        if self.config.mova_use_torch_rms_norm:
            grouped = F.rms_norm(grouped, (self.group_size,), eps=self.eps)
        else:
            grouped = grouped * torch.rsqrt(
                grouped.square().mean(dim=-1, keepdim=True) + self.eps
            )
        normalized = grouped.reshape_as(hidden_states)
        # xLLM forms ``weight + 1`` in the activation dtype before its fused
        # kernel promotes the product.  Keeping that rounding point is needed
        # for checkpoint-level BF16 parity; adding in FP32 measurably changes
        # outputs for trained offset weights.
        effective_weight = self.weight.to(input_dtype) + torch.ones(
            (), dtype=input_dtype, device=self.weight.device
        )
        return (normalized * effective_weight.float()).to(input_dtype)


class SoftplusGatedSelfAttention(SelfAttention):
    """Standard self-attention with the xLLM post-attention gate."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.config.flash_decode:
            raise ValueError("Softplus-gated attention does not support flash_decode")

    def _apply_output_gate(self, x: Tensor, gate: Tensor) -> Tensor:
        if self.config.mova_attention_gate_function == "softplus":
            gate = F.softplus(gate, beta=math.log(2.0))
        elif self.config.mova_attention_gate_function == "silu":
            gate = F.silu(gate)
        else:
            raise ValueError(
                f"Unsupported attention gate: {self.config.mova_attention_gate_function}"
            )
        return x * gate.reshape_as(x)


@dataclass
class SequentialMoVAValueExpertsSubmodules:
    """Submodules used by the correctness-first value-expert backend."""

    linear: Union[ModuleSpec, type] = None


class SequentialMoVAValueExperts(MegatronModule):
    """Value experts evaluated in expert-major order.

    This backend deliberately favors an auditable parameter layout and exact
    autograd behavior.  The grouped-GEMM backend implements the same forward
    API with a kernel-native packed parameter layout.
    """

    def __init__(
        self,
        config: MoVATransformerConfig,
        submodules: SequentialMoVAValueExpertsSubmodules,
        input_size: int,
        output_size: int,
        num_experts: int,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        super().__init__(config=config)
        self.tp_group = pg_collection.tp
        self.tensor_parallel_input = False
        self.output_size = output_size
        self.output_size_per_partition = divide(output_size, get_pg_size(self.tp_group))

        # Inputs are gathered once by MoVAValueProjection.  Disabling sequence
        # parallelism here prevents every expert projection from gathering the
        # same tokens again.
        expert_config = copy.copy(config)
        expert_config.sequence_parallel = False
        self.experts = torch.nn.ModuleList(
            [
                build_module(
                    submodules.linear,
                    input_size,
                    output_size,
                    config=expert_config,
                    init_method=config.init_method,
                    bias=False,
                    gather_output=False,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_comm_buffer_name=f"mova_value_{expert_id}",
                    tp_group=self.tp_group,
                )
                for expert_id in range(num_experts)
            ]
        )

    def forward(self, permuted_tokens: Tensor, tokens_per_expert: Tensor) -> Tensor:
        token_chunks = torch.split(permuted_tokens, tokens_per_expert.tolist(), dim=0)
        outputs = [expert(tokens)[0] for expert, tokens in zip(self.experts, token_chunks)]
        return torch.cat(outputs, dim=0)


class GroupedGemmMoVAValueExperts(MegatronModule):
    """All routed value projections in one autograd-capable grouped GEMM.

    The production tensor-parallel layout matches xLLM: each rank stores an
    input-dimension shard of every expert and produces partial full-width value
    projections. ``MoVAValueProjection`` sums and shards those projections with
    a last-dimension reduce-scatter before applying SiLU.
    """

    def __init__(
        self,
        config: MoVATransformerConfig,
        submodules: SequentialMoVAValueExpertsSubmodules,
        input_size: int,
        output_size: int,
        num_experts: int,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        del submodules
        super().__init__(config=config)
        grouped_gemm.assert_grouped_gemm_is_available()
        self.tp_group = pg_collection.tp
        self.tensor_parallel_input = True
        self.input_size = input_size
        self.num_experts = num_experts
        self.output_size = output_size
        self.tp_rank = get_pg_rank(self.tp_group)
        self.tp_size = get_pg_size(self.tp_group)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = divide(output_size, self.tp_size)

        if config.init_model_with_meta_device:
            device = "meta"
        elif config.use_cpu_initialization:
            device = "cpu"
        else:
            device = torch.cuda.current_device()
        self.weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                self.input_size_per_partition,
                output_size,
                device=device,
                dtype=config.params_dtype,
            )
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize TP-sharded experts, including after meta materialization."""
        if not hasattr(self.weight, "tensor_model_parallel"):
            set_tensor_model_parallel_attributes(self.weight, True, 1, 1)
        setattr(self.weight, "allreduce", True)

        if not self.config.perform_initialization or self.weight.is_meta:
            return

        if self.weight.device.type == "cpu":
            # Each stored expert is [input_per_tp, output]. Initialize its
            # [output, input_per_tp] transpose like RowParallelLinear.
            for expert_id in range(self.num_experts):
                expert_weight = self.weight.data[expert_id]
                _initialize_affine_weight_cpu(
                    expert_weight.transpose(0, 1),
                    self.output_size,
                    self.input_size,
                    self.input_size_per_partition,
                    partition_dim=1,
                    init_method=self.config.init_method,
                    params_dtype=self.config.params_dtype,
                    rank=self.tp_rank,
                    world_size=self.tp_size,
                    skip_set_tensor_parallel_attributes=True,
                )
        elif self.weight.device.type == "cuda":
            for expert_id in range(self.num_experts):
                expert_weight = self.weight.data[expert_id]
                _initialize_affine_weight_gpu(
                    expert_weight.transpose(0, 1),
                    self.config.init_method,
                    partition_dim=1,
                )
        else:
            raise RuntimeError(f"Unsupported MoVA value-expert device: {self.weight.device}")

    def forward(self, permuted_tokens: Tensor, tokens_per_expert: Tensor) -> Tensor:
        if not self.config.bf16:
            raise ValueError("The grouped_gemm MoVA backend currently requires bf16")
        if permuted_tokens.numel() == 0:
            return permuted_tokens.new_empty((0, self.output_size)) + self.weight.sum() * 0.0
        return grouped_gemm.ops.gmm(permuted_tokens, self.weight, tokens_per_expert, trans_b=False)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"weight": 1},
            sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


@dataclass
class MoVAValueProjectionSubmodules:
    """Submodules for routed value projection."""

    experts: Union[ModuleSpec, type] = None


class MoVAValueProjection(MegatronModule):
    """Route tokens, evaluate selected value experts, and combine outputs."""

    def __init__(
        self,
        config: MoVATransformerConfig,
        submodules: MoVAValueProjectionSubmodules,
        input_size: int,
        output_size: int,
        layer_number: int,
        pg_collection: ProcessGroupCollection,
    ) -> None:
        super().__init__(config=config)
        if config.mova_num_value_experts <= 0:
            raise ValueError("MoVAValueProjection requires at least one value expert")
        self.tp_group = pg_collection.tp
        self.tp_size = get_pg_size(self.tp_group)
        if self.tp_size > 1 and not config.sequence_parallel:
            raise ValueError("MoVA with tensor parallelism currently requires sequence_parallel")

        router_config = copy.copy(config)
        router_config.num_moe_experts = config.mova_num_value_experts
        router_config.moe_router_topk = config.mova_router_topk
        router_config.moe_router_score_function = config.mova_router_score_function
        router_config.moe_router_topk_scaling_factor = config.mova_router_topk_scaling_factor
        router_config.moe_router_enable_expert_bias = config.mova_router_enable_expert_bias
        router_config.moe_router_bias_update_rate = config.mova_router_bias_update_rate
        router_config.moe_aux_loss_coeff = config.mova_router_aux_loss_coeff
        router_config.moe_router_load_balancing_type = config.mova_router_load_balancing_type
        # xLLM uses a weight-only router plus a non-parameter selection-bias buffer.
        router_config.add_bias_linear = False

        self.router = TopKRouter(config=router_config, pg_collection=pg_collection)
        self.router.set_layer_number(layer_number)
        self.experts = build_module(
            submodules.experts,
            config=config,
            input_size=input_size,
            output_size=output_size,
            num_experts=config.mova_num_value_experts,
            pg_collection=pg_collection,
        )
        self.output_size_per_partition = self.experts.output_size_per_partition
        self.tensor_parallel_input = self.experts.tensor_parallel_input

    def forward(self, hidden_states: Tensor) -> Tensor:
        # Route sequence-parallel tokens locally so router weight gradients have
        # Megatron's standard sequence-parallel reduction semantics.
        routing_probs, routing_map = self.router(hidden_states)

        if self.tp_size > 1:
            if self.tensor_parallel_input:
                # [tokens/TP, hidden] -> [tokens, hidden/TP]. This is the
                # deployed xLLM layout and avoids replicating the full hidden
                # state (and its dgrad) for every routed value projection.
                global_shape = (hidden_states.shape[0] * self.tp_size, *hidden_states.shape[1:-1])
                hidden_states = all_to_all_sp2hp(hidden_states, group=self.tp_group)
                hidden_states = hidden_states.view(*global_shape, hidden_states.shape[-1])
            else:
                hidden_states = gather_from_sequence_parallel_region(
                    hidden_states, tensor_parallel_output_grad=False, group=self.tp_group
                )
            # Routing is computed once on local sequence-parallel tokens. Its
            # small outputs are gathered to align with the full token order.
            # Probability gradients reduce-scatter contributions from all
            # output shards; the non-differentiable map only needs a gather.
            routing_probs = gather_from_sequence_parallel_region(
                routing_probs, tensor_parallel_output_grad=True, group=self.tp_group
            )
            with torch.no_grad():
                routing_map = gather_from_sequence_parallel_region(
                    routing_map, tensor_parallel_output_grad=False, group=self.tp_group
                )

        original_shape = hidden_states.shape[:-1]
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_probs = routing_probs.reshape(-1, routing_probs.shape[-1])
        flat_map = routing_map.reshape(-1, routing_map.shape[-1])

        # grouped_gemm follows the standard MCore MoE contract and consumes
        # expert counts on CPU.  The sequential reference accepts the same
        # representation, keeping both backends behind one projection API.
        tokens_per_expert = flat_map.sum(dim=0).long().cpu()
        permuted_tokens, permuted_probs, sorted_indices = permute(
            flat_hidden,
            flat_map,
            probs=flat_probs,
            num_out_tokens=flat_hidden.shape[0] * self.config.mova_router_topk,
            fused=False,
        )
        projected = self.experts(permuted_tokens, tokens_per_expert)
        if self.tp_size > 1 and self.tensor_parallel_input:
            projected = reduce_scatter_last_dim_to_tensor_parallel_region(
                projected, group=self.tp_group
            )
        projected = F.silu(projected)
        projected = projected * permuted_probs.to(projected.dtype).unsqueeze(-1)
        mixed_values = unpermute(
            projected,
            sorted_indices,
            restore_shape=torch.Size((flat_hidden.shape[0], self.output_size_per_partition)),
        )
        return mixed_values.view(*original_shape, self.output_size_per_partition)


@dataclass
class MoVASelfAttentionSubmodules:
    """Submodules used by MoVA self-attention."""

    linear_qkg: Union[ModuleSpec, type] = None
    value_projection: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = IdentityOp
    k_layernorm: Union[ModuleSpec, type] = IdentityOp


class MoVASelfAttention(Attention):
    """Causal GQA whose values are produced by routed projections."""

    def __init__(
        self,
        config: MoVATransformerConfig,
        submodules: MoVASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )
        if not config.attention_output_gate:
            raise ValueError("The reference MoVA architecture requires attention_output_gate")
        if config.fused_single_qkv_rope:
            raise ValueError("MoVA does not support fused_single_qkv_rope")
        if config.flash_decode:
            raise ValueError("MoVA does not support flash_decode")

        query_heads_per_group = config.num_attention_heads // config.num_query_groups
        self.qkg_projection_size = (
            config.num_query_groups
            * (2 * query_heads_per_group + 1)
            * self.hidden_size_per_attention_head
        )
        self.linear_qkg = build_module(
            submodules.linear_qkg,
            config.hidden_size,
            self.qkg_projection_size,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            # QKG is the attention input column projection and therefore uses
            # Transformer Engine's standard QKV sequence-parallel user buffer.
            tp_comm_buffer_name="qkv",
            tp_group=pg_collection.tp,
        )
        self.value_projection = build_module(
            submodules.value_projection,
            config=config,
            input_size=config.hidden_size,
            output_size=self.kv_projection_size,
            layer_number=layer_number,
            pg_collection=pg_collection,
        )
        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=self.hidden_size_per_attention_head,
            config=config,
            eps=config.layernorm_epsilon,
        )
        self.k_layernorm = build_module(
            submodules.k_layernorm,
            hidden_size=self.hidden_size_per_attention_head,
            config=config,
            eps=config.layernorm_epsilon,
        )

    def get_query_key_value_tensors(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        output_gate: bool = True,
        split_qkv: bool = True,
    ):
        if key_value_states is not None:
            raise ValueError("MoVA only implements self-attention")
        if not split_qkv:
            raise ValueError("MoVA requires split Q/K/V tensors")
        if not output_gate:
            raise ValueError("The reference MoVA architecture requires its output gate")

        mixed_qkg, _ = self.linear_qkg(hidden_states)
        query_heads_per_group = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        per_group_size = (2 * query_heads_per_group + 1) * self.hidden_size_per_attention_head
        mixed_qkg = mixed_qkg.view(
            *mixed_qkg.shape[:-1], self.num_query_groups_per_partition, per_group_size
        )
        query, gate, key = torch.split(
            mixed_qkg,
            [
                query_heads_per_group * self.hidden_size_per_attention_head,
                query_heads_per_group * self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ],
            dim=-1,
        )
        query = query.reshape(*query.shape[:2], self.num_attention_heads_per_partition, -1)
        gate = gate.reshape(*gate.shape[:2], self.num_attention_heads_per_partition, -1)
        value = self.value_projection(hidden_states).view(
            *key.shape[:2], self.num_query_groups_per_partition, -1
        )

        query = self.q_layernorm(query)
        key = self.k_layernorm(key)
        return query, key, value, gate

    def _apply_output_gate(self, x: Tensor, gate: Tensor) -> Tensor:
        if self.config.mova_attention_gate_function == "softplus":
            gate = F.softplus(gate, beta=math.log(2.0))
        elif self.config.mova_attention_gate_function == "silu":
            gate = F.silu(gate)
        else:
            raise ValueError(
                f"Unsupported attention gate: {self.config.mova_attention_gate_function}"
            )
        return x * gate.reshape_as(x)

    def backward_dw(self):
        for module in (self.linear_qkg, self.linear_proj):
            if hasattr(module, "backward_dw"):
                module.backward_dw()
        value_experts = self.value_projection.experts
        if hasattr(value_experts, "experts"):
            for expert in value_experts.experts:
                if hasattr(expert, "backward_dw"):
                    expert.backward_dw()
        elif hasattr(value_experts, "backward_dw"):
            value_experts.backward_dw()
