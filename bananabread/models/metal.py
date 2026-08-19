"""Compatibility helpers for Transformers' Apple Silicon Metal quantization."""

from types import MethodType
from typing import Any


def _batched_metal_forward(self, inputs):
    """Run Metal's 2-D affine matmul over arbitrary leading dimensions."""
    from transformers.integrations.metal_quantization import MetalLinear

    if inputs.ndim <= 2:
        return MetalLinear.forward(self, inputs)

    leading_shape = inputs.shape[:-1]
    flattened = inputs.reshape(-1, inputs.shape[-1])
    output = MetalLinear.forward(self, flattened)
    return output.reshape(*leading_shape, output.shape[-1])


def enable_batched_metal_linears(model: Any) -> int:
    """Make each MetalLinear accept transformer-style 3-D batched inputs.

    Transformers 5.15's Metal kernel path passes the full 3-D tensor to a
    matrix-multiply kernel that currently produces non-finite values after the
    first batch row. Linear projection is independent across leading
    dimensions, so flattening them before the kernel and restoring them after
    it is equivalent and keeps batching enabled.
    """
    from transformers.integrations.metal_quantization import MetalLinear

    count = 0
    for module in model.modules():
        if isinstance(module, MetalLinear):
            module.forward = MethodType(_batched_metal_forward, module)
            count += 1
    return count
