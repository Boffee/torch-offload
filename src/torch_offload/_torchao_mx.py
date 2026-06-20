"""Internal optional-import module for TorchAO MX (microscaling) support.

Single source of truth for the TorchAO ``MXTensor`` layout this repo
needs to move OCP-microscaling weights through :class:`PinnedParam` and
to expose the dequantize/requantize adapter capability. TorchAO's public
workflow creates ``MXTensor`` weights via ``quantize_(...)`` with an MX
inference config (or directly through ``MXTensor.to_mx``); the adapter
only preserves, moves, and (for LoRA merge) re-encodes those
already-quantized tensors.

Scope is intentionally limited to the two element dtypes seen in real
models: MXFP8 (``float8_e4m3fn`` / ``float8_e5m2``) and MXFP4
(``float4_e2m1fn_x2``). MXFP6 and any other MX element dtype are not
admitted; such tensors fall through to a clear "no adapter" error rather
than being silently mishandled.
"""

from __future__ import annotations

from typing import Any

import torch

LAYOUT_ATTRS = (
    "qdata",
    "scale",
    "elem_dtype",
    "block_size",
    "orig_dtype",
    "kernel_preference",
    "act_quant_kwargs",
    "is_swizzled_scales",
)
"""Attributes this repo reads from a TorchAO ``MXTensor``."""


try:
    from torchao.prototype.mx_formats.mx_tensor import MXTensor

    TORCHAO_MX_AVAILABLE = True
except ImportError:
    TORCHAO_MX_AVAILABLE = False
    MXTensor: Any = None


# MX element dtypes this adapter supports. MXFP8 (e4m3/e5m2) is always
# present; MXFP4's packed dtype exists only on torch builds new enough to
# carry it, so it is probed by name. MXFP6 is deliberately excluded.
_FP4_ELEM_DTYPE = getattr(torch, "float4_e2m1fn_x2", None)
_SUPPORTED_ELEM_DTYPES: tuple[torch.dtype, ...] = tuple(
    dt
    for dt in (torch.float8_e4m3fn, torch.float8_e5m2, _FP4_ELEM_DTYPE)
    if dt is not None
)


def is_supported_mx_elem_dtype(elem_dtype: object) -> bool:
    """Return whether ``elem_dtype`` is an MX variant this repo handles."""
    return elem_dtype in _SUPPORTED_ELEM_DTYPES


def is_mx_tensor(t: object) -> bool:
    """Return whether ``t`` is a supported TorchAO MX tensor.

    A real ``MXTensor`` of an unsupported element dtype (notably MXFP6)
    returns ``False`` so it does not dispatch to this adapter.
    """
    return (
        TORCHAO_MX_AVAILABLE
        and isinstance(t, MXTensor)
        and is_supported_mx_elem_dtype(getattr(t, "elem_dtype", None))
    )


def require_mx_tensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated TorchAO MX tensor, or raise."""
    if not is_mx_tensor(t):
        raise TypeError(
            f"expected a supported TorchAO MXTensor "
            f"({_supported_elem_dtype_names()}), got {type(t).__name__}"
        )
    validate_layout(t)
    return t


def create_mx_tensor(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    elem_dtype: object,
    block_size: int,
    orig_dtype: torch.dtype,
    kernel_preference: object,
    act_quant_kwargs: object | None,
    is_swizzled_scales: bool,
) -> torch.Tensor:
    """Rebuild a TorchAO ``MXTensor`` from raw storage + metadata."""
    if not TORCHAO_MX_AVAILABLE:
        raise RuntimeError("torchao is required to create an MXTensor")
    return MXTensor(
        qdata,
        scale,
        elem_dtype,
        block_size,
        orig_dtype,
        kernel_preference,
        act_quant_kwargs,
        is_swizzled_scales,
    )


def validate_layout(t: torch.Tensor) -> None:
    """Raise if ``t`` is missing the MX attributes we preserve."""
    missing = [a for a in LAYOUT_ATTRS if not hasattr(t, a)]
    if not missing:
        return
    raise RuntimeError(
        f"MXTensor is missing expected attributes {missing!r}; "
        f"this repo is pinned to a layout that exposes {LAYOUT_ATTRS}. "
        "TorchAO likely refactored the wrapper class — upgrade "
        "torch-offload to match."
    )


def dequantize_mx_tensor(t: torch.Tensor) -> torch.Tensor:
    """Return the dense logical value of an MX tensor as fp32."""
    mx = require_mx_tensor(t)
    return mx.dequantize(mx.orig_dtype).to(device=mx.device, dtype=torch.float32)


def requantize_mx_tensor(
    t: torch.Tensor, *, like: torch.Tensor,
) -> torch.Tensor:
    """Encode dense ``t`` using the MX layout and metadata from ``like``.

    Goes through the public ``MXTensor.to_mx``, which recomputes the
    power-of-two (E8M0) block scales from the new values — so a LoRA merge
    that grows a block's amax is absorbed by a larger shared exponent.
    Element dtype, block size, original dtype, kernel preference,
    activation-quant kwargs, and the swizzled-scale layout carry over from
    ``like``.

    ``MXTensor`` does not store its weight ``ScaleCalculationMode`` on the
    wrapper, but the dynamic-activation recipe records it on
    ``act_quant_kwargs`` (one mode configures both weight and activation —
    e.g. RCEIL for the default inference config). Recover it so the merge
    re-encodes with the same scale-rounding policy the weight was
    quantized with; weight-only MX carries no kwargs, so ``to_mx``'s
    default (FLOOR) is used.
    """
    mx = require_mx_tensor(like)
    if tuple(t.shape) != tuple(mx.shape):
        raise ValueError(
            f"Cannot requantize tensor with shape {tuple(t.shape)} like "
            f"MXTensor with shape {tuple(mx.shape)}."
        )
    if not mx.qdata.is_contiguous():
        # A transposed/strided MX weight has a packed layout the re-encode
        # (which always produces the standard contiguous packing) can
        # neither consume nor fill — to_mx rejects non-contiguous input and
        # the block partition would not line up. Reject early with an
        # actionable error rather than an opaque kernel assertion.
        raise ValueError(
            "Cannot merge LoRA into a non-contiguous (e.g. transposed) MX "
            "weight: requantization produces the standard packed layout, "
            "which cannot fill a transposed target. Use routed LoRA for "
            "this weight."
        )
    scaling_mode = getattr(mx.act_quant_kwargs, "scaling_mode", None)
    scaling_mode_kwarg = {} if scaling_mode is None else {"scaling_mode": scaling_mode}
    return MXTensor.to_mx(
        t.to(dtype=mx.orig_dtype),
        mx.elem_dtype,
        block_size=mx.block_size,
        kernel_preference=mx.kernel_preference,
        act_quant_kwargs=mx.act_quant_kwargs,
        is_swizzled_scales=mx.is_swizzled_scales,
        **scaling_mode_kwarg,
    )


def _supported_elem_dtype_names() -> str:
    return ", ".join(str(dt) for dt in _SUPPORTED_ELEM_DTYPES)
