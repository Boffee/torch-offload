"""Internal optional-import module for ``optimum-quanto`` support.

Single source of truth for everything this repo depends on from
optimum-quanto:

- The ``from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor``
  optional import and the ``QUANTO_AVAILABLE`` flag.
- :data:`LAYOUT_ATTRS` — the private-attr names this repo reads on a
  ``WeightQBytesTensor``.
- :func:`dequantize_qbytes_tensor` and
  :func:`requantize_qbytes_tensor` — the pieces used by
  :class:`~torch_offload.quanto_adapter.QuantoAdapter` to expose a
  dequantize/requantize adapter capability.

Both pin/move/wrap and dequantize/requantize support consume from here
through :mod:`quanto_adapter`, so the layout assumption only has to be
updated once when optimum-quanto refactors.

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

    The streaming adapter (:meth:`QuantoAdapter.matches`) calls this so
    a layout drift in optimum-quanto is reported uniformly rather than
    as a generic ``AttributeError`` partway through the dispatch. The
    check itself is four ``hasattr`` calls — cheap to run on every
    dispatch, no caching.
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


def dequantize_qbytes_tensor(qt: torch.Tensor) -> torch.Tensor:
    """Return the dense logical value in the wrapper's compute dtype."""
    qbytes = require_qbytes_tensor(qt)
    return qbytes.dequantize()


def requantize_qbytes_tensor(
    t: torch.Tensor, *, like: torch.Tensor,
) -> torch.Tensor:
    """Encode dense ``t`` using the quanto layout and scale from ``like``."""
    qbytes = require_qbytes_tensor(like)
    if tuple(t.shape) != tuple(qbytes.size()):
        raise ValueError(
            f"Cannot requantize tensor with shape {tuple(t.shape)} like "
            f"WeightQBytesTensor with shape {tuple(qbytes.size())}."
        )
    scale = qbytes._scale.to(device=t.device).clone()
    return create_qbytes_tensor(
        qbytes.qtype, qbytes.axis, qbytes.size(), qbytes.stride(),
        _quantize_to_qbytes(t, qbytes, scale),
        scale,
        qbytes_activation_qtype(qbytes),
    )


def _quantize_to_qbytes(
    float_data: torch.Tensor,
    reference: Any,  # noqa: ANN401
    scale: torch.Tensor,
) -> torch.Tensor:
    """Quantize float data using the same scale as ``reference``."""
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
