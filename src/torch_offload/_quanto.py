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

import torch

try:
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

    QUANTO_AVAILABLE = True
except ImportError:
    QUANTO_AVAILABLE = False
    WeightQBytesTensor = None  # type: ignore[assignment,misc]


LAYOUT_ATTRS = ("_data", "_scale", "qtype", "axis")
"""Attributes this repo reads from a ``WeightQBytesTensor``.

If optimum-quanto refactors and any of these vanishes, callers that
go through :func:`validate_layout` get a framed :class:`RuntimeError`
naming the missing attribute(s); callers that don't would otherwise
fail with a generic ``AttributeError`` later in the access path.
"""


def validate_layout(qt: torch.Tensor) -> None:
    """Raise if ``qt`` is missing any of :data:`LAYOUT_ATTRS`.

    Cheap one-time guard. Both the streaming adapter
    (:meth:`QuantoAdapter.matches`) and the permanent-merge path
    (:func:`requantize_with_addmm_delta`) call this so a layout drift
    in optimum-quanto is reported uniformly rather than as a generic
    ``AttributeError`` partway through the dispatch.
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
    validate_layout(qt)
    dev = qt.device
    float_data = qt.dequantize().to(device=dev, dtype=torch.float32)
    float_data.addmm_(
        b.to(device=dev, dtype=torch.float32),
        a.to(device=dev, dtype=torch.float32),
        alpha=strength,
    )
    return WeightQBytesTensor.create(  # type: ignore[union-attr]
        qt.qtype, qt.axis, qt.size(), qt.stride(),
        _quantize_to_qbytes(float_data, qt),
        qt._scale.clone(),
        getattr(qt, "activation_qtype", None),
    )


def _quantize_to_qbytes(
    float_data: torch.Tensor, reference: torch.Tensor,
) -> torch.Tensor:
    """Quantize float data using the same scale as ``reference``."""
    scale = reference._scale
    axis = reference.axis
    if axis == 0:
        scaled = float_data / scale.view(-1, *([1] * (float_data.dim() - 1)))
    else:
        scaled = float_data / scale
    return scaled.round().clamp(
        torch.iinfo(reference._data.dtype).min,
        torch.iinfo(reference._data.dtype).max,
    ).to(reference._data.dtype)
