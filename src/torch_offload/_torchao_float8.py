"""Internal optional-import module for TorchAO scaled-FP8 support.

Single source of truth for the TorchAO ``Float8Tensor`` layout this
repo needs to move scaled-fp8 weights through :class:`PinnedParam` and
to expose the dequantize/requantize adapter capability. TorchAO's
public workflow creates ``Float8Tensor`` weights via
``quantize_(..., Float8WeightOnlyConfig/Float8DynamicActivation...)``;
the adapter only preserves, moves, and (for LoRA merge) re-encodes
those already-quantized tensors.
"""

from __future__ import annotations

from typing import Any

import torch

LAYOUT_ATTRS = (
    "qdata",
    "scale",
    "block_size",
    "mm_config",
    "act_quant_kwargs",
    "kernel_preference",
)
"""Attributes this repo reads from a TorchAO ``Float8Tensor``."""


try:
    from torchao.quantization.granularity import PerRow, PerTensor
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
        Float8Tensor,
    )

    TORCHAO_FLOAT8_AVAILABLE = True
except ImportError:
    TORCHAO_FLOAT8_AVAILABLE = False
    Float8Tensor: Any = None
    PerRow: Any = None
    PerTensor: Any = None


def is_float8_tensor(t: object) -> bool:
    """Return whether ``t`` is a TorchAO scaled-fp8 tensor."""
    return TORCHAO_FLOAT8_AVAILABLE and isinstance(t, Float8Tensor)


def require_float8_tensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated TorchAO Float8Tensor, or raise."""
    if not is_float8_tensor(t):
        raise TypeError(f"expected TorchAO Float8Tensor, got {type(t).__name__}")
    validate_layout(t)
    return t


def create_float8_tensor(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_size: list[int],
    mm_config: object | None,
    act_quant_kwargs: object | None,
    kernel_preference: object,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Rebuild a TorchAO ``Float8Tensor`` from raw storage + metadata."""
    if not TORCHAO_FLOAT8_AVAILABLE:
        raise RuntimeError("torchao is required to create a Float8Tensor")
    return Float8Tensor(
        qdata,
        scale,
        block_size=block_size,
        mm_config=mm_config,
        act_quant_kwargs=act_quant_kwargs,
        kernel_preference=kernel_preference,
        dtype=dtype,
    )


def validate_layout(t: torch.Tensor) -> None:
    """Raise if ``t`` is missing the Float8 attributes we preserve."""
    missing = [a for a in LAYOUT_ATTRS if not hasattr(t, a)]
    if not missing:
        return
    raise RuntimeError(
        f"Float8Tensor is missing expected attributes {missing!r}; "
        f"this repo is pinned to a layout that exposes {LAYOUT_ATTRS}. "
        "TorchAO likely refactored the wrapper class — upgrade "
        "torch-offload to match."
    )


def dequantize_float8_tensor(t: torch.Tensor) -> torch.Tensor:
    """Return the dense logical value of a scaled-fp8 tensor as fp32."""
    f8 = require_float8_tensor(t)
    return f8.dequantize().to(device=f8.device, dtype=torch.float32)


def requantize_float8_tensor(
    t: torch.Tensor, *, like: torch.Tensor,
) -> torch.Tensor:
    """Encode dense ``t`` using the fp8 layout and metadata from ``like``.

    Goes through the public ``Float8Tensor.from_hp`` so the scale is
    recomputed for the new values (a LoRA merge can grow the per-block
    amax past what ``like``'s scale covers). Granularity is recovered
    from ``like.block_size``; all dispatch metadata (mm config, kernel
    preference, activation quant kwargs, fp8 dtype) carries over.
    """
    f8 = require_float8_tensor(like)
    if tuple(t.shape) != tuple(f8.shape):
        raise ValueError(
            f"Cannot requantize tensor with shape {tuple(t.shape)} like "
            f"Float8Tensor with shape {tuple(f8.shape)}."
        )
    granularity = _granularity_from_block_size(
        tuple(f8.block_size), tuple(f8.shape),
    )
    return Float8Tensor.from_hp(
        t.to(dtype=f8.dtype),
        float8_dtype=f8.qdata.dtype,
        granularity=granularity,
        mm_config=f8.mm_config,
        kernel_preference=f8.kernel_preference,
        act_quant_kwargs=f8.act_quant_kwargs,
    )


def _granularity_from_block_size(
    block_size: tuple[int, ...], shape: tuple[int, ...],
) -> object:
    """Invert TorchAO's ``get_block_size`` for the FP8 granularities.

    ``Float8Tensor`` supports ``PerTensor`` (block covers the whole
    tensor) and ``PerRow(dim)`` (block covers axis ``dim`` fully, 1
    elsewhere). Shapes where both readings coincide (e.g. a dim of
    size 1) produce identical block partitions either way, so any
    matching reading is correct.
    """
    if block_size == shape:
        return PerTensor()
    for dim, size in enumerate(shape):
        per_row = tuple(size if i == dim else 1 for i in range(len(shape)))
        if block_size == per_row:
            return PerRow(dim=dim)
    raise ValueError(
        f"Float8Tensor block_size {block_size!r} for shape {shape!r} "
        "matches neither PerTensor nor PerRow granularity; TorchAO "
        "likely added a granularity this adapter does not support yet."
    )
