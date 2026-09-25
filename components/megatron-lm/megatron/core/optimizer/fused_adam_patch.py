"""Monkey-patch for TransformerEngine FusedAdam to support CPU allocation
during low-memory checkpoint resume (--low-memory-resume).

This prevents GPU OOM during checkpoint loading by initially allocating
optimizer states (exp_avg, exp_avg_sq) on CPU, then moving them to GPU
after the checkpoint is fully loaded.

The patch is only applied when low_memory_resume is enabled, so
_patched_initialize_state unconditionally allocates on CPU.
"""

import torch
from logging import getLogger

logger = getLogger(__name__)
_patch_applied = False


def _patched_initialize_state(self, param, state_name, zero_buffer, store_param_remainders=False):
    """Patched _initialize_state: allocate on CPU for low-memory resume."""
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
    from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
    import transformer_engine_torch as tex

    dtype = self.name_to_dtype_map[state_name]
    param_for_empty = param.dequantize() if isinstance(param, QuantizedTensor) else param

    device = torch.device('cpu')

    if store_param_remainders:
        data = torch.zeros(param_for_empty.shape, dtype=torch.int16, device=device)
    else:
        data = torch.empty(param_for_empty.shape, dtype=dtype, device=device)

    if zero_buffer:
        data.zero_()

    if dtype == torch.uint8:
        quantizer = Float8Quantizer(
            scale=torch.ones([1], dtype=torch.float32, device=device),
            amax=torch.zeros([1], dtype=torch.float32, device=device),
            fp8_dtype=tex.DType.kFloat8E4M3,
        )
        self.state[param][state_name] = quantizer.make_empty(param.shape)
        self.state[param][state_name].quantize_(data.float())
    else:
        self.state[param][state_name] = data

    if dtype != torch.float32:
        if param not in self._scales:
            self._scales[param] = {}
        self._scales[param][state_name] = torch.ones([1], dtype=torch.float32, device=device)


def apply_fused_adam_patch():
    global _patch_applied
    if _patch_applied:
        return

    from transformer_engine.pytorch.optimizers import FusedAdam
    FusedAdam._initialize_state = _patched_initialize_state
    _patch_applied = True


def is_patch_applied():
    return _patch_applied
