"""Internal optional-import module for TorchAO INT8 support.

Single source of truth for the TorchAO ``Int8Tensor`` layout this repo
needs to move int8-quantized weights through :class:`PinnedParam` and to
expose the dequantize/requantize adapter capability. TorchAO's public
workflow creates ``Int8Tensor`` weights via
``quantize_(..., Int8WeightOnlyConfig(version=2) /
Int8DynamicActivationInt8WeightConfig(version=2))``; the adapter only
preserves, moves, and (for LoRA merge) re-encodes those already-quantized
tensors.

``Int8Tensor`` carries two required storage tensors (``qdata`` int8 +
``scale``) and up to four optional ones (``zero_point`` for asymmetric
weight quant, plus ``act_quant_scale`` / ``act_quant_zero_point`` /
``act_pre_scale`` for static-activation recipes). Absent optionals are
``None`` and are skipped end-to-end by the structured-adapter base.
"""

from __future__ import annotations

from typing import Any

import torch

from ._torchao_granularity import granularity_from_block_size

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
    from torchao.quantization.quant_primitives import MappingType
    from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
        Int8Tensor,
    )

    TORCHAO_INT8_AVAILABLE = True
except ImportError:
    TORCHAO_INT8_AVAILABLE = False
    Int8Tensor: Any = None
    MappingType: Any = None


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


def dequantize_int8_tensor(t: torch.Tensor) -> torch.Tensor:
    """Return the dense logical value in the wrapper's compute dtype."""
    qt = require_int8_tensor(t)
    return qt.dequantize()


def requantize_int8_tensor(
    t: torch.Tensor, *, like: torch.Tensor,
) -> torch.Tensor:
    """Encode dense ``t`` using the int8 layout and metadata from ``like``.

    Goes through the public ``Int8Tensor.from_hp`` with ``scale=None`` so
    the per-block weight scale (and, for asymmetric quant, the zero point)
    is recomputed for the new values — a LoRA merge can grow the per-block
    amax past what ``like``'s scale covers, and int8's 256-level grid would
    otherwise clip it. Granularity is recovered from ``like.block_size``.
    All activation-quant state (static ``act_quant_scale`` /
    ``act_quant_zero_point`` / ``act_pre_scale`` and dynamic
    ``act_quant_kwargs``) carries over from ``like`` unchanged, since a
    weight-side merge does not touch activation calibration.

    ``Int8Tensor`` does not record its weight ``MappingType``, so the
    mapping type is inferred from the stored zero point: a non-trivial
    (non-zero) zero point means asymmetric, otherwise symmetric. This is
    exact for the standard symmetric recipes (whose zero point is an
    all-zero tensor) and for ordinary asymmetric weights; it would only
    misread an asymmetric weight whose zero points happen to all be zero —
    a case TorchAO's stock configs do not produce.
    """
    qt = require_int8_tensor(like)
    if tuple(t.shape) != tuple(qt.shape):
        raise ValueError(
            f"Cannot requantize tensor with shape {tuple(t.shape)} like "
            f"Int8Tensor with shape {tuple(qt.shape)}."
        )
    mapping_type = (
        MappingType.ASYMMETRIC
        if qt.zero_point is not None and bool(qt.zero_point.any())
        else MappingType.SYMMETRIC
    )
    granularity = granularity_from_block_size(
        tuple(qt.block_size), tuple(qt.shape), label="Int8Tensor",
    )
    return Int8Tensor.from_hp(
        t.to(dtype=qt.dtype),
        granularity,
        mapping_type,
        act_quant_kwargs=qt.act_quant_kwargs,
        act_quant_scale=qt.act_quant_scale,
        act_quant_zero_point=qt.act_quant_zero_point,
        act_pre_scale=qt.act_pre_scale,
    )
