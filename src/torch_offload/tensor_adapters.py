"""Tensor-type adapters: per-type pin/move/wrap mechanics.

Different tensor subclasses need different machinery to move bytes
across the CPU↔GPU boundary while preserving correctness:

- Plain ``torch.Tensor`` (bf16/fp16/fp32): single contiguous pinned
  buffer; consumers either replace ``module._parameters[leaf]`` with a
  fresh :class:`nn.Parameter` wrapping it, or ``.data``-swap to preserve
  identity for trainable params.
- Quanto ``WeightQBytesTensor``: two pinned tensors (``_data`` + ``_scale``)
  plus quant metadata; the wrapper must be reconstructed on each move.
  ``.data``-swap doesn't work for quanto — its quant state is part of
  the Parameter's wrapped object, not its bytes — so quanto stays
  frozen-only via registry replacement.

Each adapter encapsulates the mechanics for one tensor type. The rest
of the package is type-agnostic and dispatches through
``tensor_adapter_registry.select_adapter``. The base adapter contract is
intentionally small: clone/pin, move to GPU, rebuild wrappers, report
cache bytes, and report the logical compute dtype. Adapters also
provide a layout signature for block-pool compatibility checks. Extra
operations are expressed as small optional protocols so callers ask for
the exact capability they need instead of hard-coding tensor classes.

This module is internal to :mod:`torch_offload`. It contains the base
contracts and plain tensor adapter only; built-in adapter selection
lives in :mod:`tensor_adapter_registry`.
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
    "DequantRequantCopyIntoTensorAdapter",
    "DequantRequantTensorAdapter",
    "ParameterDataSwapTensorAdapter",
    "TensorAdapter",
    "TensorCopyIntoAdapter",
    "adapter_name",
    "clone_to_pinned_cpu",
]


DENSE_ADDMM_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# Adapter-specific opaque state types. The Protocol is generic over
# them so consumers (PinnedParam) can stay tensor-type-agnostic
# while each adapter pins its own concrete state shape.
PinnedStateT = TypeVar("PinnedStateT")
GpuStateT = TypeVar("GpuStateT")


@runtime_checkable
class TensorAdapter(Protocol[PinnedStateT, GpuStateT]):
    """Adapter encoding the mechanics of pinning, moving, and wrapping
    one tensor type. Adapter instances are stateless; they hold no
    per-param data.

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
        """True if this adapter handles tensor ``t``. Implementations
        should be conservative — :class:`RegularAdapter` matches only plain
        ``torch.Tensor``, not unrecognized subclasses."""
        ...

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple:
        """Composite identity key for tied-weight detection. Two tensors
        with the same key share backing data and quant metadata; different
        keys must not be deduped. Includes device and view layout
        (shape/stride/offset) so distinct devices or views into the
        same buffer don't collapse."""
        ...

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple:
        """Hashable tensor layout metadata for block-pool compatibility.

        Unlike :meth:`tensor_id`, this must not include tensor identity.
        It captures only fields that must match for one GPU
        pool target to safely receive bytes from multiple block instances.
        """
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
        Used as the deactivated-state registry value
        (``module._parameters[leaf] = cpu_param``).

        ``requires_grad`` defaults to ``False`` to match frozen
        registry-replacement callers; pass ``True`` when building a wrapper
        for trainable storage.
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
    def validate_dense_addmm_target(t: torch.Tensor) -> None:
        """Raise if ``t`` cannot safely receive an in-place ``addmm_``.

        Used before installing a post-copy hook that mutates the freshly
        copied GPU tensor. Adapters that do not implement this capability
        are treated as non-mergeable by that caller.
        """
        ...


@runtime_checkable
class DequantRequantTensorAdapter(TensorAdapter[PinnedStateT, GpuStateT], Protocol):
    """Optional capability for shape-preserving dequantize/requantize updates.

    ``dequantize(t)`` returns a dense logical tensor for ``t``. The
    adapter owns the compute dtype choice. ``requantize(t, like=...)``
    converts a shape-compatible dense tensor back into the same
    representation/layout as ``like``. Device follows the dense input
    tensor; callers can move tensors explicitly before calling.
    """

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        """Return a dense logical tensor for ``t``."""
        ...

    @staticmethod
    def requantize(t: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        """Return ``t`` encoded in the same representation as ``like``."""
        ...


@runtime_checkable
class ParameterDataSwapTensorAdapter(TensorAdapter[PinnedStateT, GpuStateT], Protocol):
    """Optional capability for trainable streaming via ``Parameter.data`` swap."""

    @staticmethod
    def validate_parameter_data_swap_target(t: torch.Tensor) -> None:
        """Raise if ``t`` cannot safely round-trip through ``param.data =``.

        Streamed trainables preserve user Parameter identity by swapping
        only ``.data``. Tensor subclasses with wrapper metadata generally
        must not opt into this capability.
        """
        ...


@runtime_checkable
class TensorCopyIntoAdapter(TensorAdapter[PinnedStateT, GpuStateT], Protocol):
    """Optional capability for representation-preserving copy into ``target``.

    ``copy_into(src, target=...)`` copies ``src``'s representation into
    ``target`` while preserving ``target``'s object identity and storage.
    Structured tensor wrappers use this when generic ``target.copy_(src)``
    does not update their internal storage correctly.
    """

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        """Copy ``src`` into ``target``'s existing representation."""
        ...


@runtime_checkable
class DequantRequantCopyIntoTensorAdapter(
    DequantRequantTensorAdapter[PinnedStateT, GpuStateT],
    TensorCopyIntoAdapter[PinnedStateT, GpuStateT],
    Protocol,
):
    """Combined capability for in-place representation-preserving updates."""


# ---------------------------------------------------------------------------
# RegularAdapter — plain torch.Tensor (bf16/fp16/fp32, etc.)
# ---------------------------------------------------------------------------


def clone_to_pinned_cpu(
    t: torch.Tensor,
    *,
    memory_format: torch.memory_format = torch.preserve_format,
) -> torch.Tensor:
    """Clone ``t`` into pinned CPU memory from any source device."""
    source = t.detach()
    if source.device.type == "cpu":
        return source.clone(memory_format=memory_format).pin_memory()

    if memory_format == torch.preserve_format:
        pinned = torch.empty_strided(
            tuple(source.shape),
            source.stride(),
            dtype=source.dtype,
            device="cpu",
        ).pin_memory()
    else:
        pinned = torch.empty_like(
            source,
            device="cpu",
            memory_format=memory_format,
        ).pin_memory()
    pinned.copy_(source)
    return pinned


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
    and GPU storages. Frozen model-bound callers replace the module registry via
    ``module._parameters[leaf] = ...`` with a pinned CPU wrapper or
    active GPU wrapper; trainable callers preserve Parameter identity
    by skipping registry replacement and ``.data``-swapping into their own
    persistent Parameter. Both paths are supported by the shape of this
    adapter (plain tensors round-trip through ``.data =`` cleanly).

    Conservative on dispatch: only matches exactly
    ``type(t) is torch.Tensor`` (or ``nn.Parameter``). Unrecognized
    tensor subclasses fall through to other adapters or raise via the
    factory.
    """

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        # Strict identity match on the base class. PEFT, FSDP, quanto,
        # DTensor, etc. are subclasses with extra state; a silent fallback
        # to RegularAdapter would clone-and-dequantize quanto or break
        # distributed placement. Each subclass needs its own adapter.
        return type(t) is torch.Tensor or type(t) is nn.Parameter

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple:
        return (
            "regular",
            t.device,
            t.data_ptr(),
            t.dtype,
            tuple(t.shape),
            t.stride(),
            t.storage_offset(),
        )

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple:
        return (tuple(t.shape), t.dtype)

    @staticmethod
    def clone_pin(t: torch.Tensor) -> _RegularPinned:
        return _RegularPinned(
            data=clone_to_pinned_cpu(
                t.data,
                memory_format=torch.contiguous_format,
            )
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
    def validate_dense_addmm_target(t: torch.Tensor) -> None:
        if type(t) is not torch.Tensor:
            raise ValueError(
                f"Dense addmm target is {type(t).__name__}; "
                "dense in-place addmm requires a plain torch.Tensor."
            )
        if t.dtype not in DENSE_ADDMM_DTYPES:
            raise ValueError(
                f"Dense addmm target has dtype {t.dtype}; "
                "dense in-place addmm requires bf16, fp16, or fp32."
            )

    @staticmethod
    def validate_parameter_data_swap_target(t: torch.Tensor) -> None:
        if type(t) is not torch.Tensor:
            raise NotImplementedError(
                f"Parameter data-swap target is {type(t).__name__}; "
                "Parameter.data swap requires a plain torch.Tensor."
            )

    @staticmethod
    def cache_bytes(state: _RegularPinned) -> int:
        return state.data.numel() * state.data.element_size()


def adapter_name(adapter: TensorAdapter[Any, Any]) -> str:
    """Human-readable name for an adapter instance."""
    return type(adapter).__name__
