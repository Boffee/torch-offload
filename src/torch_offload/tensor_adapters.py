"""Tensor-type adapters: per-type pin/move/wrap mechanics.

Different tensor subclasses need different machinery to move bytes
across the CPU↔GPU boundary while preserving correctness:

- Plain ``torch.Tensor`` (bf16/fp16/fp32): single contiguous pinned
  buffer; consumers either slot-replace via ``module._parameters[leaf]
  = ...`` with a fresh :class:`nn.Parameter` wrapping it, or ``.data``-
  swap to preserve identity for trainable params.
- Quanto ``WeightQBytesTensor``: two pinned tensors (``_data`` + ``_scale``)
  plus quant metadata; the wrapper must be reconstructed on each move.
  ``.data``-swap doesn't work for quanto — its quant state is part of
  the Parameter's wrapped object, not its bytes — so quanto stays
  frozen-only via slot replacement.

Each adapter encapsulates the mechanics for one tensor type. The rest
of the package (:class:`PinnedParamBuffer`, :class:`PinnedWeights`,
:class:`StreamedWeights`) is type-agnostic and dispatches through
:func:`select_adapter`. The base adapter contract is intentionally small:
clone/pin, move to GPU, rebuild wrappers, report cache bytes, and report
the logical compute dtype. Extra operations are expressed as small
optional protocols so callers ask for the exact capability they need
instead of hard-coding tensor classes.

This module is internal to :mod:`torch_offload`. Adapters are registered
at module import time; new types can be added by writing a new adapter
class and calling :func:`register_adapter`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

import torch
from torch import nn

__all__ = [
    "DENSE_ADDMM_DTYPES",
    "CpuRoundTripTensorAdapter",
    "DenseAddmmTensorAdapter",
    "ParameterDataSwapTensorAdapter",
    "TensorAdapter",
    "register_adapter",
    "select_adapter",
]


DENSE_ADDMM_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# Adapter-specific opaque state types. The Protocol is generic over
# them so consumers (PinnedParamBuffer) can stay tensor-type-agnostic
# while each adapter pins its own concrete state shape.
PinnedStateT = TypeVar("PinnedStateT")
GpuStateT = TypeVar("GpuStateT")


@runtime_checkable
class TensorAdapter(Protocol[PinnedStateT, GpuStateT]):
    """Adapter encoding the mechanics of pinning, moving, and wrapping
    one tensor type. Adapters are stateless; they hold no per-param data.

    Generic over two opaque state types: ``PinnedStateT`` (the pinned
    host representation) and ``GpuStateT`` (the GPU storage). Each
    adapter pins these to its own concrete dataclasses; consumers
    round-trip the opaque types without inspecting them.

    The Protocol is methods-only — capability is determined by what an
    adapter implements, not by declarative flags. If a workload needs an
    operation beyond inference movement, it should check one of the
    smaller capability protocols below.
    """

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        """True if this adapter handles tensor ``t``. Used by
        :func:`select_adapter` for dispatch. Implementations should be
        conservative — :class:`RegularAdapter` matches only plain
        ``torch.Tensor``, not unrecognized subclasses."""
        ...

    @staticmethod
    def storage_key(t: torch.Tensor) -> tuple:
        """Composite identity key for tied-weight detection. Two tensors
        with the same key share storage and quant metadata; different
        keys must not be deduped. Includes view layout (shape/stride/
        offset) so distinct views into the same buffer don't collapse."""
        ...

    @staticmethod
    def clone_pin(t: torch.Tensor) -> PinnedStateT:
        """Clone ``t`` into pinned (or regular) host memory. Returns
        opaque adapter-specific state used by subsequent operations."""
        ...

    @staticmethod
    def cpu_param(
        state: PinnedStateT, *, requires_grad: bool = False
    ) -> nn.Parameter:
        """Build a stable :class:`nn.Parameter` wrapping the host state.
        Used as the deactivated-state slot value
        (``module._parameters[leaf] = cpu_param``).

        ``requires_grad`` defaults to ``False`` to match the historic
        frozen-only callers; pass ``True`` when building a wrapper for
        a trainable param the caller intends to keep in the model tree.
        """
        ...

    @staticmethod
    def alloc_gpu(state: PinnedStateT, device: torch.device) -> GpuStateT:
        """Allocate empty GPU storage mirroring this state's layout.
        Returns opaque adapter-specific state."""
        ...

    @staticmethod
    def gpu_param(
        pinned: PinnedStateT, gpu_state: GpuStateT, *, requires_grad: bool = False
    ) -> nn.Parameter:
        """Build a stable :class:`nn.Parameter` wrapping the GPU state.
        Reused across many :meth:`copy_to_gpu` calls.

        Takes both the pinned host state and the GPU state because
        adapters with structured tensors (e.g. quanto) need metadata
        captured at pin time to reconstruct the GPU-side wrapper. Plain
        adapters ignore ``pinned``.

        ``requires_grad`` defaults to ``False``; pass ``True`` for
        trainable use cases where the wrapper participates in autograd.
        """
        ...

    @staticmethod
    def copy_to_gpu(
        src: PinnedStateT, dst: GpuStateT, *, non_blocking: bool = False
    ) -> None:
        """Bulk DMA the pinned state's bytes into pre-allocated GPU storage."""
        ...

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        """Return the logical compute dtype for operations using ``t``.

        For plain tensors this is simply ``t.dtype``. Quantized wrappers
        should return their logical matmul/output dtype, not necessarily
        the dtype of packed inner storage.
        """
        ...

    @staticmethod
    def cache_bytes(state: PinnedStateT) -> int:
        """Total bytes this state consumes in host memory. Used by
        :class:`ModelCache` for budget accounting."""
        ...


@runtime_checkable
class CpuRoundTripTensorAdapter(TensorAdapter[PinnedStateT, GpuStateT], Protocol):
    """Optional D2H counterpart to the base H2D movement contract."""

    @staticmethod
    def copy_to_cpu(
        src: GpuStateT, dst: PinnedStateT, *, non_blocking: bool = False
    ) -> None:
        """Bulk D2H the GPU state's bytes into pinned host storage.

        Symmetric counterpart to :meth:`copy_to_gpu`. Used to sync the
        pinned host clone with post-update GPU contents — e.g., after
        an optimizer step has written into the GPU param, scatter the
        update back to the pinned state so the next H2D reads it.

        Adapters whose GPU representation is not round-trippable should
        not implement this capability.
        """
        ...


@runtime_checkable
class DenseAddmmTensorAdapter(TensorAdapter[PinnedStateT, GpuStateT], Protocol):
    """Optional capability for in-place dense ``addmm_`` updates."""

    @staticmethod
    def validate_dense_addmm_target(t: torch.Tensor, name: str) -> None:
        """Raise if ``t`` cannot safely receive an in-place ``addmm_``.

        Used before installing a post-copy hook that mutates the freshly
        copied GPU tensor. Adapters that do not implement this capability
        are treated as non-mergeable by that caller.
        """
        ...


@runtime_checkable
class ParameterDataSwapTensorAdapter(TensorAdapter[PinnedStateT, GpuStateT], Protocol):
    """Optional capability for trainable streaming via ``Parameter.data`` swap."""

    @staticmethod
    def validate_parameter_data_swap_target(t: torch.Tensor, name: str) -> None:
        """Raise if ``t`` cannot safely round-trip through ``param.data =``.

        Streamed trainables preserve user Parameter identity by swapping
        only ``.data``. Tensor subclasses with wrapper metadata generally
        must not opt into this capability.
        """
        ...


# ---------------------------------------------------------------------------
# RegularAdapter — plain torch.Tensor (bf16/fp16/fp32, etc.)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _RegularPinned:
    """Pinned-CPU state for a regular tensor: one contiguous host buffer."""

    data: torch.Tensor


@dataclass(slots=True)
class _RegularGpu:
    """GPU state for a regular tensor: one contiguous device buffer."""

    data: torch.Tensor


class RegularAdapter:
    """Adapter for plain ``torch.Tensor`` (no subclass machinery).

    Builds fresh :class:`nn.Parameter` objects wrapping the pinned-CPU
    and GPU storages. The frozen-only callers (:class:`PinnedWeights`,
    ``_BlockPinnedStore``) slot-replace via ``module._parameters[leaf]
    = ...`` with the buffer's ``cpu_param`` or its pool-slot
    ``gpu_param``; the user's original Parameter object is orphaned,
    so optimizer state keyed on the pre-wrap object is lost. Trainable
    callers can either request ``requires_grad=True`` wrappers or skip
    the slot replacement entirely and ``.data``-swap into their own
    persistent Parameter — both are supported by the shape of this
    adapter (plain tensors round-trip through ``.data =`` cleanly).

    Conservative on dispatch: only matches exactly
    ``type(t) is torch.Tensor`` (or ``nn.Parameter``). Unrecognized
    tensor subclasses fall through to other adapters or raise via
    :func:`select_adapter`.
    """

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        # Strict identity match on the base class. PEFT, FSDP, quanto,
        # DTensor, etc. are subclasses with extra state; a silent fallback
        # to RegularAdapter would clone-and-dequantize quanto or break
        # distributed placement. Each subclass needs its own adapter.
        return type(t) is torch.Tensor or type(t) is nn.Parameter

    @staticmethod
    def storage_key(t: torch.Tensor) -> tuple:
        return (
            "regular",
            t.data_ptr(),
            t.dtype,
            tuple(t.shape),
            t.stride(),
            t.storage_offset(),
        )

    @staticmethod
    def clone_pin(t: torch.Tensor) -> _RegularPinned:
        return _RegularPinned(
            data=t.data.clone(memory_format=torch.contiguous_format).pin_memory()
        )

    @staticmethod
    def cpu_param(
        state: _RegularPinned, *, requires_grad: bool = False
    ) -> nn.Parameter:
        return nn.Parameter(state.data, requires_grad=requires_grad)

    @staticmethod
    def alloc_gpu(state: _RegularPinned, device: torch.device) -> _RegularGpu:
        return _RegularGpu(data=torch.empty_like(state.data, device=device))

    @staticmethod
    def gpu_param(
        pinned: _RegularPinned,
        gpu_state: _RegularGpu,
        *,
        requires_grad: bool = False,
    ) -> nn.Parameter:
        _ = pinned
        # pinned unused: regular tensors carry no metadata beyond storage.
        # Argument kept for Protocol parity with TensorAdapter — quanto
        # and other structured tensors need it to reconstruct wrappers.
        return nn.Parameter(gpu_state.data, requires_grad=requires_grad)

    @staticmethod
    def copy_to_gpu(
        src: _RegularPinned, dst: _RegularGpu, *, non_blocking: bool = False
    ) -> None:
        dst.data.copy_(src.data, non_blocking=non_blocking)

    @staticmethod
    def copy_to_cpu(
        src: _RegularGpu, dst: _RegularPinned, *, non_blocking: bool = False
    ) -> None:
        dst.data.copy_(src.data, non_blocking=non_blocking)

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        return t.dtype

    @staticmethod
    def validate_dense_addmm_target(t: torch.Tensor, name: str) -> None:
        if type(t) is not torch.Tensor:
            raise ValueError(
                f"Dense addmm target {name!r} is {type(t).__name__}; "
                "dense in-place addmm requires a plain torch.Tensor."
            )
        if t.dtype not in DENSE_ADDMM_DTYPES:
            raise ValueError(
                f"Dense addmm target {name!r} has dtype {t.dtype}; "
                "dense in-place addmm requires bf16, fp16, or fp32."
            )

    @staticmethod
    def validate_parameter_data_swap_target(t: torch.Tensor, name: str) -> None:
        if type(t) is not torch.Tensor:
            raise NotImplementedError(
                f"Trainable streaming slot {name!r} is {type(t).__name__}; "
                "Parameter.data swap requires a plain torch.Tensor."
            )

    @staticmethod
    def cache_bytes(state: _RegularPinned) -> int:
        return state.data.numel() * state.data.element_size()


# ---------------------------------------------------------------------------
# Adapter registry / dispatch
# ---------------------------------------------------------------------------

# Adapters are tried in registration order. The first whose ``matches()``
# returns True wins. RegularAdapter is appended last as the conservative
# fallback for plain tensors. Subclass adapters (quanto, etc.) register
# themselves at import time and slot in front.
_ADAPTERS: list[type[TensorAdapter[Any, Any]]] = []


def register_adapter(adapter: type[TensorAdapter[Any, Any]]) -> None:
    """Register an adapter for use by :func:`select_adapter`. Adapters
    registered later take priority over earlier ones for ``matches()``
    dispatch — this lets specialized adapters (quanto, FP8 variants)
    precede :class:`RegularAdapter`."""
    if adapter not in _ADAPTERS:
        _ADAPTERS.insert(0, adapter)


def select_adapter(t: torch.Tensor) -> type[TensorAdapter[Any, Any]]:
    """Find the registered adapter that handles tensor ``t``.

    Tries adapters in reverse registration order (newest first), returning
    the first whose :meth:`TensorAdapter.matches` returns True. Raises
    :class:`NotImplementedError` if no adapter matches — the alpha library
    refuses to silently dequantize or otherwise mishandle unknown tensor
    subclasses.
    """
    for adapter in _ADAPTERS:
        if adapter.matches(t):
            return adapter
    raise NotImplementedError(
        f"No registered TensorAdapter for tensor type {type(t).__name__!r}. "
        f"Plain tensors are handled by RegularAdapter; tensor subclasses "
        f"need a dedicated adapter (see optimum.quanto integration in "
        f"quanto_adapter.py for an example)."
    )


# Register the regular fallback last so subclass adapters take priority.
register_adapter(RegularAdapter)
