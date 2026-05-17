"""Internal optional-import module for TorchAO NVFP4 support.

Single source of truth for the private-ish TorchAO NVFP4 layout this
repo needs to move packed weights through :class:`PinnedParamBuffer`.
TorchAO's public workflow creates ``NVFP4Tensor`` weights via
``quantize_(..., NVFP4WeightOnlyConfig/NVFP4DynamicActivation...)``;
the adapter only preserves and moves those already-quantized tensors.
"""

from __future__ import annotations

from typing import Any

import torch

LAYOUT_ATTRS = (
    "qdata",
    "scale",
    "block_size",
    "orig_dtype",
    "per_tensor_scale",
    "act_per_tensor_scale",
    "is_swizzled_scales",
    "use_triton_kernel",
    "act_quant_kwargs",
)
"""Attributes this repo reads from a TorchAO ``NVFP4Tensor``."""


try:
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    TORCHAO_NVFP4_AVAILABLE = True
except ImportError:
    TORCHAO_NVFP4_AVAILABLE = False
    NVFP4Tensor: Any = None


def is_nvfp4_tensor(t: object) -> bool:
    """Return whether ``t`` is a TorchAO NVFP4 tensor."""
    return TORCHAO_NVFP4_AVAILABLE and isinstance(t, NVFP4Tensor)


def require_nvfp4_tensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated TorchAO NVFP4 tensor, or raise."""
    if not is_nvfp4_tensor(t):
        raise TypeError(f"expected TorchAO NVFP4Tensor, got {type(t).__name__}")
    validate_layout(t)
    return t


def create_nvfp4_tensor(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_size: int,
    orig_dtype: torch.dtype,
    per_tensor_scale: torch.Tensor | None,
    act_per_tensor_scale: torch.Tensor | None,
    is_swizzled_scales: bool,
    use_triton_kernel: bool,
    act_quant_kwargs: object | None,
) -> torch.Tensor:
    """Rebuild a TorchAO ``NVFP4Tensor`` from raw storage + metadata."""
    if not TORCHAO_NVFP4_AVAILABLE:
        raise RuntimeError("torchao is required to create an NVFP4Tensor")
    return NVFP4Tensor(
        qdata=qdata,
        scale=scale,
        block_size=block_size,
        orig_dtype=orig_dtype,
        per_tensor_scale=per_tensor_scale,
        act_per_tensor_scale=act_per_tensor_scale,
        is_swizzled_scales=is_swizzled_scales,
        use_triton_kernel=use_triton_kernel,
        act_quant_kwargs=act_quant_kwargs,
    )


def validate_layout(t: torch.Tensor) -> None:
    """Raise if ``t`` is missing the NVFP4 attributes we preserve."""
    missing = [a for a in LAYOUT_ATTRS if not hasattr(t, a)]
    if not missing:
        return
    raise RuntimeError(
        f"NVFP4Tensor is missing expected attributes {missing!r}; "
        f"this repo is pinned to a layout that exposes {LAYOUT_ATTRS}. "
        "TorchAO likely refactored the wrapper class — upgrade "
        "torch-offload to match."
    )
