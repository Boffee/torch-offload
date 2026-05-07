"""Quanto :class:`WeightQBytesTensor` adapter.

Quanto-quantized weights are subclassed tensors that wrap multiple
internal tensors (``_data``, ``_scale``) plus quant metadata
(``qtype``, ``axis``, ``activation_qtype``). The wrapper does not
support ``p.data = ...`` storage swap — its quant state is part of the
Parameter's wrapped object, not its bytes. So this adapter:

- Decomposes ``WeightQBytesTensor`` into ``_data`` and ``_scale``,
  pins each separately.
- Reconstructs a fresh ``WeightQBytesTensor`` (and thus a fresh
  :class:`nn.Parameter`) on each activate via slot replacement.
  PyTorch optimizers keyed by the user's pre-wrap Parameter id are
  orphaned across cycles — quanto-quantized weights are inference-only.

Reaches into quanto's private attributes (``_data``, ``_scale``,
``qtype``, ``axis``, ``activation_qtype``). Pinned to the
``WeightQBytesTensor`` layout in optimum-quanto as of the version this
repo depends on. If quanto refactors the wrapper class, the
:class:`QuantoAdapter` will fail with a clear validation error at
:meth:`matches` (validates the expected attributes exist on first
match).

Auto-registers when this module is imported. Importing fails silently
if optimum-quanto is not installed — quanto support is optional.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ._quanto import QUANTO_AVAILABLE, WeightQBytesTensor, validate_layout
from .tensor_adapters import register_adapter


@dataclass(slots=True)
class _QuantoPinned:
    """Pinned-CPU state for a quanto tensor: two pinned tensors
    plus the quant metadata needed to reconstruct the wrapper."""

    data: torch.Tensor   # pinned int8/fp8
    scale: torch.Tensor  # pinned fp16/fp32
    qtype: object
    axis: object
    size: torch.Size
    stride: tuple
    act_qt: object


@dataclass(slots=True)
class _QuantoGpu:
    """GPU state for a quanto tensor: the two GPU tensors. Quant
    metadata lives in the originating :class:`_QuantoPinned`; only
    storage moves to GPU."""

    data: torch.Tensor
    scale: torch.Tensor


def _build_qbytes(
    state: _QuantoPinned, data: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Reconstruct a :class:`WeightQBytesTensor` from raw pieces +
    cached quant metadata."""
    return WeightQBytesTensor.create(  # type: ignore[union-attr]
        state.qtype, state.axis, state.size, state.stride,
        data, scale, state.act_qt,
    )


class QuantoAdapter:
    """Adapter for ``optimum.quanto.WeightQBytesTensor``.

    Decompose-on-pin, reconstruct-on-move. Each activate creates a
    fresh ``WeightQBytesTensor`` and a fresh :class:`nn.Parameter`,
    installed via slot replacement. This breaks PyTorch optimizer
    references — quanto-quantized weights are inference-only.
    """

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        if not QUANTO_AVAILABLE or not isinstance(t, WeightQBytesTensor):
            return False
        # Validate the layout we read in clone_pin/_build_qbytes is
        # still present. Cheap (four hasattr calls) and runs on every
        # dispatch — no caching needed at this scale.
        validate_layout(t)
        return True

    @staticmethod
    def storage_key(t: torch.Tensor) -> tuple:
        # Composite identity: the two underlying buffers plus the quant
        # metadata. Two distinct WeightQBytesTensors with the same
        # underlying _data/_scale storage AND matching quant params are
        # the same logical tensor and dedup safely.
        return (
            "quanto",
            t._data.data_ptr(),
            t._data.dtype,
            tuple(t._data.shape),
            t._data.stride(),
            t._data.storage_offset(),
            t._scale.data_ptr(),
            t._scale.dtype,
            tuple(t._scale.shape),
            t._scale.stride(),
            t._scale.storage_offset(),
            t.qtype,
            t.axis,
            tuple(t.size()),
            t.stride(),
            getattr(t, "activation_qtype", None),
        )

    @staticmethod
    def clone_pin(t: torch.Tensor) -> _QuantoPinned:
        # contiguous_format clone: fp8-quanto leaves some _data buffers
        # strided via internal transposes; the default preserve_format
        # would carry that through pin_memory(), breaking downstream
        # assumptions of a contiguous pinned buffer. The original quant
        # stride is captured separately and reapplied via
        # WeightQBytesTensor.create on the GPU side.
        return _QuantoPinned(
            data=t._data.clone(memory_format=torch.contiguous_format).pin_memory(),
            scale=t._scale.clone(memory_format=torch.contiguous_format).pin_memory(),
            qtype=t.qtype,
            axis=t.axis,
            size=t.size(),
            stride=t.stride(),
            act_qt=getattr(t, "activation_qtype", None),
        )

    @staticmethod
    def cpu_param(state: _QuantoPinned) -> nn.Parameter:
        qt = _build_qbytes(state, state.data, state.scale)
        return nn.Parameter(qt, requires_grad=False)

    @staticmethod
    def alloc_gpu(state: _QuantoPinned, device: torch.device) -> _QuantoGpu:
        return _QuantoGpu(
            data=torch.empty_like(state.data, device=device),
            scale=torch.empty_like(state.scale, device=device),
        )

    @staticmethod
    def gpu_param(pinned: _QuantoPinned, gpu_state: _QuantoGpu) -> nn.Parameter:
        # Quant metadata comes from the pinned state; only the storage
        # tensors come from the GPU side.
        qt = _build_qbytes(pinned, gpu_state.data, gpu_state.scale)
        return nn.Parameter(qt, requires_grad=False)

    @staticmethod
    def copy_to_gpu(
        src: _QuantoPinned, dst: _QuantoGpu, *, non_blocking: bool = False
    ) -> None:
        dst.data.copy_(src.data, non_blocking=non_blocking)
        dst.scale.copy_(src.scale, non_blocking=non_blocking)

    @staticmethod
    def cache_bytes(state: _QuantoPinned) -> int:
        return (
            state.data.numel() * state.data.element_size()
            + state.scale.numel() * state.scale.element_size()
        )


if QUANTO_AVAILABLE:
    register_adapter(QuantoAdapter)
