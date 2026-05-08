"""Internal optional-import module for ``optimum-quanto`` support.

Single source of truth for everything this repo depends on from
optimum-quanto:

- The ``from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor``
  optional import and the ``QUANTO_AVAILABLE`` flag.
- :data:`LAYOUT_ATTRS` — the private-attr names this repo reads on a
  ``WeightQBytesTensor``.
- :func:`requantize_with_addmm_delta` — the dequant→addmm→requant cycle
  used by :func:`~torch_offload.merge.merge_lora` to apply a permanent
  LoRA delta to a quantized weight.

Both :mod:`quanto_adapter` (pin/move/wrap during streaming) and
:mod:`merge` (permanent merge) consume from here so the layout assumption
only has to be updated once when optimum-quanto refactors.

Pinned to optimum-quanto's internal layout. Not part of the public API.
"""

from __future__ import annotations

from typing import Any

import torch

LAYOUT_ATTRS = ("_data", "_scale", "qtype", "axis")
"""Attributes this repo reads from a ``WeightQBytesTensor``.

If optimum-quanto refactors and any of these vanishes, callers that
go through :func:`validate_layout` get a framed :class:`RuntimeError`
naming the missing attribute(s); callers that don't would otherwise
fail with a generic ``AttributeError`` later in the access path.
"""


try:
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

    QUANTO_AVAILABLE = True
except ImportError:
    QUANTO_AVAILABLE = False
    WeightQBytesTensor: Any = None


def is_weight_qbytes_tensor(t: object) -> bool:
    """Return whether ``t`` is an optimum-quanto WeightQBytesTensor."""
    return QUANTO_AVAILABLE and isinstance(t, WeightQBytesTensor)


def require_qbytes_tensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated quanto tensor, or raise."""
    if not is_weight_qbytes_tensor(t):
        raise TypeError(f"expected optimum-quanto WeightQBytesTensor, got {type(t).__name__}")
    validate_layout(t)
    return t


def qbytes_activation_qtype(t: Any) -> object | None:  # noqa: ANN401
    """Optional activation quant type stored by some quanto versions."""
    return getattr(t, "activation_qtype", None)


def create_qbytes_tensor(
    qtype: object,
    axis: int,
    size: torch.Size,
    stride: tuple[int, ...],
    data: torch.Tensor,
    scale: torch.Tensor,
    activation_qtype: object | None,
) -> torch.Tensor:
    """Create an optimum-quanto WeightQBytesTensor."""
    if not QUANTO_AVAILABLE:
        raise RuntimeError("optimum-quanto is required to create a WeightQBytesTensor")
    return WeightQBytesTensor.create(qtype, axis, size, stride, data, scale, activation_qtype)


def validate_layout(qt: torch.Tensor) -> None:
    """Raise if ``qt`` is missing any of :data:`LAYOUT_ATTRS`.

    Both the streaming adapter (:meth:`QuantoAdapter.matches`) and the
    permanent-merge path (:func:`requantize_with_addmm_delta`) call
    this so a layout drift in optimum-quanto is reported uniformly
    rather than as a generic ``AttributeError`` partway through the
    dispatch. The check itself is four ``hasattr`` calls — cheap to
    run on every dispatch, no caching.
    """
    missing = [a for a in LAYOUT_ATTRS if not hasattr(qt, a)]
    if not missing:
        return
    raise RuntimeError(
        f"WeightQBytesTensor is missing expected attributes {missing!r}; "
        f"this repo is pinned to a layout that exposes {LAYOUT_ATTRS}. "
        "optimum-quanto likely refactored the wrapper class — upgrade "
        "torch-offload to match."
    )


def requantize_with_addmm_delta(
    qt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Apply ``W += strength * B @ A`` to a quanto-quantized weight.

    Dequantizes to fp32, applies the addmm, requantizes using the
    original scale, and returns a fresh :class:`WeightQBytesTensor`.
    Lossy but standard for permanent LoRA merges into quantized bases.
    """
    qbytes = require_qbytes_tensor(qt)
    dev = qbytes.device
    float_data = qbytes.dequantize().to(device=dev, dtype=torch.float32)
    float_data.addmm_(
        b.to(device=dev, dtype=torch.float32),
        a.to(device=dev, dtype=torch.float32),
        alpha=strength,
    )
    return create_qbytes_tensor(
        qbytes.qtype, qbytes.axis, qbytes.size(), qbytes.stride(),
        _quantize_to_qbytes(float_data, qbytes),
        qbytes._scale.clone(),
        qbytes_activation_qtype(qbytes),
    )


def _quantize_to_qbytes(
    float_data: torch.Tensor, reference: Any,  # noqa: ANN401
) -> torch.Tensor:
    """Quantize float data using the same scale as ``reference``."""
    scale = reference._scale
    axis = reference.axis
    scaled = (
        float_data / scale.view(-1, *([1] * (float_data.dim() - 1)))
        if axis == 0
        else float_data / scale
    )
    return scaled.round().clamp(
        torch.iinfo(reference._data.dtype).min,
        torch.iinfo(reference._data.dtype).max,
    ).to(reference._data.dtype)
