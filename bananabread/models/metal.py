"""Compatibility helpers for Transformers' Apple Silicon Metal quantization."""

from types import MethodType
from typing import Any
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_metal_kernel():
    """Use the v2 ABI explicitly; Transformers 5.19 still requests v1."""
    from transformers.integrations.hub_kernels import get_kernel

    return get_kernel("kernels-community/mlx-quantization-metal-kernels", version=2)


def _batched_metal_forward(self, inputs):
    """Run Metal's 2-D affine matmul over arbitrary leading dimensions."""
    import torch
    from transformers.integrations.metal_quantization import MetalLinear

    if self.weight.dtype != torch.uint32:
        return MetalLinear.forward(self, inputs)

    leading_shape = inputs.shape[:-1]
    flattened = inputs.reshape(-1, inputs.shape[-1])
    output = _get_metal_kernel().affine_qmm_t(
        flattened,
        self.weight,
        self.scales.to(inputs.dtype),
        self.qbiases.to(inputs.dtype),
        self.group_size,
        self.bits,
    )
    if self.bias is not None:
        output = output + self.bias
    return output.reshape(*leading_shape, output.shape[-1])


def enable_batched_metal_linears(model: Any) -> int:
    """Make each MetalLinear accept transformer-style 3-D batched inputs.

    Transformers' Metal kernel path passes the full 3-D tensor to a
    matrix-multiply kernel that currently produces non-finite values after the
    first batch row. Linear projection is independent across leading
    dimensions, so flattening them before the kernel and restoring them after
    it is equivalent and keeps batching enabled. Quantized layers use kernel
    v2 directly, avoiding Transformers 5.19's hardcoded v1 request.
    """
    from transformers.integrations.metal_quantization import MetalLinear

    count = 0
    for module in model.modules():
        if isinstance(module, MetalLinear):
            module.forward = MethodType(_batched_metal_forward, module)
            count += 1
    return count
