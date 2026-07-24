"""Internal optional-import module for TorchAO static-activation FP8.

TorchAO's prototype static FP8 workflow represents a weight with
``PrototypeFloat8Tensor``.  In addition to the ordinary FP8 weight bytes and
weight scale, the wrapper owns the calibrated activation scale used by stock
``nn.Linear`` dispatch.  This module is the single source of truth for the
layout that torch-offload preserves and for the generic
dequantize/requantize adapter capability.

TorchAO 0.17 requires a supplied static scale to have the same rank as the
activation passed to ``PrototypeFloat8Tensor.from_hp``.  A checkpoint scalar
therefore cannot natively serve both a 2-D and a 3-D Linear input.  At import
time we narrowly wrap PrototypeFloat8Tensor's Linear handlers: calibrated
per-tensor activations are flattened to 2-D, quantized with a rank-2 view of
the same one-element scale, and reshaped back afterwards.  No tensor is
mutated, and non-static/non-scalar PrototypeFloat8Tensor paths retain their
original TorchAO handlers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from ._torchao_granularity import granularity_from_block_size

LAYOUT_ATTRS = (
    "qdata",
    "scale",
    "act_quant_scale",
    "block_size",
    "mm_config",
    "act_quant_kwargs",
    "kernel_preference",
)
"""Attributes this repo reads from a static ``PrototypeFloat8Tensor``."""


try:
    from torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor import (
        PrototypeFloat8Tensor,
    )
    from torchao.quantization.granularity import PerTensor
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
        QuantizeTensorToFloat8Kwargs,
    )

    TORCHAO_STATIC_FLOAT8_AVAILABLE = True
except ImportError:
    TORCHAO_STATIC_FLOAT8_AVAILABLE = False
    PrototypeFloat8Tensor: Any = None
    QuantizeTensorToFloat8Kwargs: Any = None
    PerTensor: Any = None


_LINEAR_PATCH_ERROR: str | None = None
_LinearHandler = Callable[
    [object, tuple[type, ...], tuple[object, ...], dict[str, object]],
    torch.Tensor,
]


def _rank2_static_weight(t: Any) -> torch.Tensor:  # noqa: ANN401
    """Alias ``t`` with its scalar activation scale viewed as rank two."""
    return PrototypeFloat8Tensor(
        t.qdata,
        t.scale,
        act_quant_scale=t.act_quant_scale.reshape(1, 1),
        block_size=list(t.block_size),
        mm_config=t.mm_config,
        act_quant_kwargs=t.act_quant_kwargs,
        kernel_preference=t.kernel_preference,
        dtype=t.dtype,
    )


def _run_rank_agnostic_linear(
    original: _LinearHandler,
    func: object,
    types: tuple[type, ...],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> torch.Tensor:
    """Delegate a Linear call, flattening calibrated scalar activations."""
    input_tensor, weight_tensor = args[0], args[1]
    act_quant_scale = getattr(weight_tensor, "act_quant_scale", None)
    output_scale = getattr(weight_tensor, "output_act_quant_scale", None)
    output_kwargs = getattr(weight_tensor, "output_act_quant_kwargs", None)
    if (
        not isinstance(input_tensor, torch.Tensor)
        or isinstance(input_tensor, PrototypeFloat8Tensor)
        or not isinstance(weight_tensor, PrototypeFloat8Tensor)
        or act_quant_scale is None
        or act_quant_scale.numel() != 1
        or input_tensor.ndim == 0
        # Output-static quantization is a separate Prototype feature and is
        # not part of this adapter's qdata/scale/act_quant_scale contract.
        or output_scale is not None
        or output_kwargs is not None
    ):
        return original(func, types, args, kwargs)

    input_shape = tuple(input_tensor.shape)
    flattened = input_tensor.reshape(-1, input_shape[-1])
    weight_rank2 = _rank2_static_weight(weight_tensor)
    result = original(
        func,
        types,
        (flattened, weight_rank2, *args[2:]),
        kwargs,
    )
    return result.reshape(*input_shape[:-1], result.shape[-1])


def _install_rank_agnostic_linear() -> None:
    """Wrap TorchAO's ATen and torch-function Linear dispatch once."""
    if getattr(PrototypeFloat8Tensor, "_torch_offload_linear_patch", False):
        return

    aten_table = PrototypeFloat8Tensor._ATEN_OP_TABLE[PrototypeFloat8Tensor]
    torch_fn_table = PrototypeFloat8Tensor._TORCH_FN_TABLE[PrototypeFloat8Tensor]
    aten_linear = torch.ops.aten.linear.default
    original_aten = aten_table[aten_linear]
    original_functional = torch_fn_table[F.linear]

    def aten_handler(
        func: object,
        types: tuple[type, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> torch.Tensor:
        return _run_rank_agnostic_linear(
            original_aten, func, types, args, kwargs
        )

    def functional_handler(
        func: object,
        types: tuple[type, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> torch.Tensor:
        return _run_rank_agnostic_linear(
            original_functional, func, types, args, kwargs
        )

    aten_table[aten_linear] = aten_handler
    torch_fn_table[F.linear] = functional_handler
    PrototypeFloat8Tensor._torch_offload_linear_patch = True


if TORCHAO_STATIC_FLOAT8_AVAILABLE:
    try:
        _install_rank_agnostic_linear()
    except (AttributeError, KeyError) as exc:
        # Keep importing torch-offload optional when TorchAO refactors its
        # prototype dispatch.  A matching tensor receives a precise layout
        # error from validate_layout instead of breaking package import.
        _LINEAR_PATCH_ERROR = f"{type(exc).__name__}: {exc}"


def is_static_float8_tensor(t: object) -> bool:
    """Return whether ``t`` is a calibrated PrototypeFloat8Tensor."""
    return (
        TORCHAO_STATIC_FLOAT8_AVAILABLE
        and isinstance(t, PrototypeFloat8Tensor)
        and getattr(t, "act_quant_scale", None) is not None
    )


def require_static_float8_tensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated static PrototypeFloat8Tensor, or raise."""
    if not is_static_float8_tensor(t):
        raise TypeError(
            "expected a static TorchAO PrototypeFloat8Tensor with "
            f"act_quant_scale, got {type(t).__name__}"
        )
    validate_layout(t)
    return t


def create_static_float8_tensor(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    act_quant_scale: torch.Tensor,
    block_size: list[int],
    mm_config: object | None,
    act_quant_kwargs: object,
    kernel_preference: object,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Rebuild a static PrototypeFloat8Tensor from storage + metadata."""
    if not TORCHAO_STATIC_FLOAT8_AVAILABLE:
        raise RuntimeError(
            "torchao is required to create a static PrototypeFloat8Tensor"
        )
    return PrototypeFloat8Tensor(
        qdata,
        scale,
        act_quant_scale=act_quant_scale,
        block_size=block_size,
        mm_config=mm_config,
        act_quant_kwargs=act_quant_kwargs,
        kernel_preference=kernel_preference,
        dtype=dtype,
    )


def validate_layout(t: torch.Tensor) -> None:
    """Raise unless ``t`` is the supported static per-tensor FP8 layout."""
    missing = [attr for attr in LAYOUT_ATTRS if not hasattr(t, attr)]
    if missing:
        raise RuntimeError(
            "PrototypeFloat8Tensor is missing expected attributes "
            f"{missing!r}; this repo is pinned to a layout that exposes "
            f"{LAYOUT_ATTRS}. TorchAO likely refactored the prototype "
            "wrapper class — upgrade torch-offload to match."
        )
    wrapped: Any = t
    if _LINEAR_PATCH_ERROR is not None:
        raise RuntimeError(
            "PrototypeFloat8Tensor's Linear dispatch layout is incompatible "
            f"with this torch-offload version ({_LINEAR_PATCH_ERROR})."
        )
    if not isinstance(wrapped.act_quant_scale, torch.Tensor):
        raise TypeError("PrototypeFloat8Tensor.act_quant_scale must be a tensor")
    if wrapped.act_quant_scale.numel() != 1:
        raise ValueError(
            "StaticFloat8Adapter supports a per-tensor activation scale "
            "containing exactly one value."
        )
    if not isinstance(wrapped.act_quant_kwargs, QuantizeTensorToFloat8Kwargs):
        raise ValueError(
            "StaticFloat8Adapter requires Float8 activation quantization "
            "metadata on PrototypeFloat8Tensor.act_quant_kwargs."
        )
    if not isinstance(wrapped.act_quant_kwargs.granularity, PerTensor):
        raise ValueError(
            "StaticFloat8Adapter supports only per-tensor static activation "
            "quantization."
        )
    if tuple(wrapped.block_size) != tuple(wrapped.shape):
        raise ValueError(
            "StaticFloat8Adapter supports only the per-tensor FP8 weight "
            "layout used by TorchAO's static workflow; use Float8Tensor's "
            "dynamic activation path for other weight granularities."
        )
    if (
        getattr(wrapped, "output_act_quant_scale", None) is not None
        or getattr(wrapped, "output_act_quant_kwargs", None) is not None
    ):
        raise ValueError(
            "StaticFloat8Adapter does not support PrototypeFloat8Tensor "
            "output activation quantization."
        )


def dequantize_static_float8_tensor(t: torch.Tensor) -> torch.Tensor:
    """Return the dense logical value in the wrapper's compute dtype."""
    f8 = require_static_float8_tensor(t)
    return f8.dequantize()


def requantize_static_float8_tensor(
    t: torch.Tensor, *, like: torch.Tensor
) -> torch.Tensor:
    """Re-encode weight values while preserving activation calibration.

    The weight scale is intentionally recomputed because a LoRA merge can
    increase the weight range.  ``act_quant_scale`` is weight-independent
    calibration state, so it is attached unchanged to the new wrapper.
    """
    f8 = require_static_float8_tensor(like)
    if tuple(t.shape) != tuple(f8.shape):
        raise ValueError(
            f"Cannot requantize tensor with shape {tuple(t.shape)} like "
            "static PrototypeFloat8Tensor with shape "
            f"{tuple(f8.shape)}."
        )
    granularity = granularity_from_block_size(
        tuple(f8.block_size),
        tuple(f8.shape),
        label="PrototypeFloat8Tensor",
    )
    out = PrototypeFloat8Tensor.from_hp(
        t.to(dtype=f8.dtype),
        float8_dtype=f8.qdata.dtype,
        granularity=granularity,
        mm_config=f8.mm_config,
        kernel_preference=f8.kernel_preference,
        act_quant_kwargs=f8.act_quant_kwargs,
        act_quant_scale=f8.act_quant_scale.to(device=t.device),
    )
    return _repair_zero_scale_blocks(out)


def _repair_zero_scale_blocks(f8: Any) -> torch.Tensor:  # noqa: ANN401
    """Replace TorchAO's 0/0 encoding for an all-zero weight block."""
    zero = f8.scale == 0
    if not bool(zero.any()):
        return f8
    eps = torch.finfo(torch.float32).eps
    scale = torch.where(zero, torch.full_like(f8.scale, eps), f8.scale)
    qdata = torch.where(zero, torch.zeros_like(f8.qdata), f8.qdata)
    return create_static_float8_tensor(
        qdata,
        scale,
        f8.act_quant_scale,
        list(f8.block_size),
        f8.mm_config,
        f8.act_quant_kwargs,
        f8.kernel_preference,
        f8.dtype,
    )
