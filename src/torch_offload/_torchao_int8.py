"""Internal optional-import module for TorchAO INT8 support.

Single source of truth for the TorchAO ``Int8Tensor`` layout this repo
needs to move int8-quantized weights through :class:`PinnedParam`.
TorchAO's public workflow creates ``Int8Tensor`` weights via
``quantize_(..., Int8WeightOnlyConfig(version=2) /
Int8DynamicActivationInt8WeightConfig(version=2))``; the adapter only
preserves and moves those already-quantized tensors.

``Int8Tensor`` carries two required storage tensors (``qdata`` int8 +
``scale``) and up to four optional ones (``zero_point`` for asymmetric
weight quant, plus ``act_quant_scale`` / ``act_quant_zero_point`` /
``act_pre_scale`` for static-activation recipes). Absent optionals are
``None`` and are skipped end-to-end by the structured-adapter base.
"""

from __future__ import annotations

from typing import Any

import torch

LAYOUT_ATTRS = (
    "qdata",
    "scale",
    "zero_point",
    "act_quant_scale",
    "act_quant_zero_point",
    "act_pre_scale",
    "block_size",
    "dtype",
    "act_quant_kwargs",
)
"""Attributes this repo reads from a TorchAO ``Int8Tensor``."""


try:
    from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
        Int8Tensor,
    )

    TORCHAO_INT8_AVAILABLE = True
except ImportError:
    TORCHAO_INT8_AVAILABLE = False
    Int8Tensor: Any = None


def is_int8_tensor(t: object) -> bool:
    """Return whether ``t`` is a TorchAO ``Int8Tensor``."""
    return TORCHAO_INT8_AVAILABLE and isinstance(t, Int8Tensor)


def require_int8_tensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated TorchAO Int8Tensor, or raise."""
    if not is_int8_tensor(t):
        raise TypeError(f"expected TorchAO Int8Tensor, got {type(t).__name__}")
    validate_layout(t)
    return t


def create_int8_tensor(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_size: list[int],
    dtype: torch.dtype,
    zero_point: torch.Tensor | None,
    act_quant_scale: torch.Tensor | None,
    act_quant_zero_point: torch.Tensor | None,
    act_pre_scale: torch.Tensor | None,
    act_quant_kwargs: object | None,
) -> torch.Tensor:
    """Rebuild a TorchAO ``Int8Tensor`` from raw storage + metadata."""
    if not TORCHAO_INT8_AVAILABLE:
        raise RuntimeError("torchao is required to create an Int8Tensor")
    return Int8Tensor(
        qdata,
        scale,
        block_size,
        dtype,
        zero_point=zero_point,
        act_quant_scale=act_quant_scale,
        act_quant_zero_point=act_quant_zero_point,
        act_pre_scale=act_pre_scale,
        act_quant_kwargs=act_quant_kwargs,
    )


def validate_layout(t: torch.Tensor) -> None:
    """Raise if ``t`` is missing the Int8 attributes we preserve."""
    missing = [a for a in LAYOUT_ATTRS if not hasattr(t, a)]
    if not missing:
        return
    raise RuntimeError(
        f"Int8Tensor is missing expected attributes {missing!r}; "
        f"this repo is pinned to a layout that exposes {LAYOUT_ATTRS}. "
        "TorchAO likely refactored the wrapper class — upgrade "
        "torch-offload to match."
    )
