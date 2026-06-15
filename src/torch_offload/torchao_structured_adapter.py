"""Shared base for TorchAO structured-tensor adapters.

TorchAO represents each quantized weight as a tensor subclass wrapping a
handful of inner storage tensors (packed elements + scales) plus opaque
dispatch metadata. Moving such a weight across the CPU<->GPU boundary is
the same mechanical dance for every format: clone each inner tensor into
pinned host memory, allocate stride-matching GPU storage, bulk-copy, and
rebuild the wrapper from storage + a metadata snapshot. Only a few things
vary per format: which inner tensors exist (and whether some are
optional), how the wrapper is reconstructed, the identity/layout metadata
that distinguishes incompatible variants, and the logical compute dtype.

:class:`TorchaoStructuredAdapter` captures the shared mechanics once and
exposes the per-format parts as small hooks. Concrete adapters
(:class:`~torch_offload.float8_adapter.Float8Adapter`,
:class:`~torch_offload.mx_adapter.MxAdapter`,
:class:`~torch_offload.nvfp4_adapter.Nvfp4Adapter`, ...) subclass it and
implement those hooks.

Capabilities beyond inference movement (CPU round-trip, dequantize/
requantize LoRA merge) are added by the concrete subclass that supports
them — and only that subclass — so the ``@runtime_checkable`` capability
protocols, which test for method presence on the class, report each
format's true capability. The base deliberately implements inference
movement only: subclasses that add nothing (MX, NVFP4) stay
frozen-inference, while a subclass that defines the extra methods
(Float8) advertises exactly those capabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

import torch
from torch import nn

from .tensor_adapters import (
    clone_to_pinned_cpu,
    empty_like_strided,
    optional_tensor_id,
    tensor_layout,
)

MetaT = TypeVar("MetaT")

__all__ = [
    "TorchaoGpu",
    "TorchaoPinned",
    "TorchaoStructuredAdapter",
    "copy_storage",
]


@dataclass(slots=True, frozen=True)
class TorchaoPinned(Generic[MetaT]):
    """Pinned-CPU state for a TorchAO structured tensor.

    ``storage`` holds the inner storage tensors positionally, parallel to
    the adapter's ``_STORAGE_NAMES``; an entry is ``None`` for an absent
    optional tensor (e.g. NVFP4 global scales). ``meta`` is the immutable
    metadata snapshot used to rebuild the wrapper on each move.
    """

    storage: tuple[torch.Tensor | None, ...]
    meta: MetaT


@dataclass(slots=True, frozen=True)
class TorchaoGpu:
    """GPU state for a TorchAO structured tensor: device storage only;
    metadata lives in the originating :class:`TorchaoPinned`."""

    storage: tuple[torch.Tensor | None, ...]


def copy_storage(
    src: tuple[torch.Tensor | None, ...],
    dst: tuple[torch.Tensor | None, ...],
    *,
    non_blocking: bool,
) -> None:
    """Per-tensor bulk copy of a parallel storage tuple, skipping absent
    (``None``) entries. Used for both H2D and the Float8 D2H round-trip."""
    for s, d in zip(src, dst, strict=True):
        if s is None:
            continue
        assert d is not None
        d.copy_(s, non_blocking=non_blocking)


class TorchaoStructuredAdapter(ABC, Generic[MetaT]):
    """Base adapter for TorchAO structured (subclass-wrapped) weights.

    Implements the full
    :class:`~torch_offload.tensor_adapters.TensorAdapter` movement and
    identity contract in terms of per-format hooks. A concrete adapter
    sets :attr:`_TAG` / :attr:`_STORAGE_NAMES` and implements the ``_*``
    hooks; it opts into capabilities beyond inference movement by defining
    the relevant methods (``copy_to_cpu``, ``dequantize`` / ``requantize``
    / ``copy_into``) itself.

    The hooks operate on the validated wrapper, typed :data:`Any` because
    TorchAO tensor subclasses are untyped — the same boundary the
    per-format ``require_*_tensor`` helpers expose.
    """

    _TAG: ClassVar[str]
    _STORAGE_NAMES: ClassVar[tuple[str, ...]]

    # --- per-format hooks -------------------------------------------------

    @staticmethod
    @abstractmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        """True if ``t`` is this format's TorchAO subclass (and a supported
        variant). Mirrors the per-format ``is_*_tensor`` predicate."""

    @staticmethod
    @abstractmethod
    def _validate_layout(t: torch.Tensor) -> None:
        """Raise if ``t`` is missing the attributes this adapter preserves."""

    @staticmethod
    @abstractmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        """Return ``t`` as the validated wrapper, or raise."""

    @staticmethod
    @abstractmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        """Live inner storage tensors, positional and parallel to
        :attr:`_STORAGE_NAMES`; ``None`` for an absent optional tensor."""

    @staticmethod
    @abstractmethod
    def _meta_of(t: Any) -> MetaT:  # noqa: ANN401
        """Snapshot the wrapper's reconstruction metadata."""

    @staticmethod
    @abstractmethod
    def _reconstruct(
        storage: tuple[torch.Tensor | None, ...], meta: MetaT
    ) -> torch.Tensor:
        """Rebuild the TorchAO wrapper from storage + metadata snapshot."""

    @staticmethod
    @abstractmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        """Format-specific identity metadata fragments for :meth:`tensor_id`."""

    @classmethod
    def _layout_metadata(cls, t: Any) -> tuple[object, ...]:  # noqa: ANN401
        """Layout metadata fragments for :meth:`layout_signature`.

        Defaults to the identity metadata — correct when block-pool
        compatibility needs exactly the fields that distinguish identity.
        Override only when layout needs fewer/different fields (e.g. Float8
        drops the logical dtype, which the standard dtype slot already
        carries)."""
        return cls._id_metadata(t)

    @staticmethod
    @abstractmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        """Logical matmul/output dtype read off the validated wrapper."""

    # --- shared mechanics -------------------------------------------------

    @classmethod
    def matches(cls, t: torch.Tensor) -> bool:
        if not cls._is_tensor(t):
            return False
        cls._validate_layout(t)
        return True

    @classmethod
    def tensor_id(cls, t: torch.Tensor) -> tuple[object, ...]:
        w = cls._require(t)
        return (
            cls._TAG,
            *(optional_tensor_id(s) for s in cls._storage_of(w)),
            tuple(w.shape),
            w.stride(),
            *cls._id_metadata(w),
        )

    @classmethod
    def layout_signature(cls, t: torch.Tensor) -> tuple[object, ...]:
        """Hashable layout metadata used by block-pool compatibility checks."""
        w = cls._require(t)
        return (
            tuple(w.shape),
            w.dtype,
            w.stride(),
            *cls._layout_metadata(w),
            *(
                (name, tensor_layout(s))
                for name, s in zip(
                    cls._STORAGE_NAMES, cls._storage_of(w), strict=True
                )
            ),
        )

    @classmethod
    def clone_pin(cls, t: torch.Tensor) -> TorchaoPinned[MetaT]:
        w = cls._require(t)
        # preserve_format (clone_to_pinned_cpu default): inner-tensor stride
        # ordering can encode a transposed quantized tensor.
        storage = tuple(
            clone_to_pinned_cpu(s) if s is not None else None
            for s in cls._storage_of(w)
        )
        return TorchaoPinned(storage=storage, meta=cls._meta_of(w))

    @classmethod
    def cpu_param(
        cls, state: TorchaoPinned[MetaT], *, requires_grad: bool = False
    ) -> nn.Parameter:
        return nn.Parameter(
            cls._reconstruct(state.storage, state.meta),
            requires_grad=requires_grad,
        )

    @classmethod
    def alloc_gpu(
        cls, state: TorchaoPinned[MetaT], device: torch.device
    ) -> TorchaoGpu:
        return TorchaoGpu(
            storage=tuple(
                empty_like_strided(s, device) if s is not None else None
                for s in state.storage
            )
        )

    @classmethod
    def gpu_param(
        cls,
        pinned: TorchaoPinned[MetaT],
        gpu_state: TorchaoGpu,
        *,
        requires_grad: bool = False,
    ) -> nn.Parameter:
        return nn.Parameter(
            cls._reconstruct(gpu_state.storage, pinned.meta),
            requires_grad=requires_grad,
        )

    @classmethod
    def copy_to_gpu(
        cls,
        src: TorchaoPinned[MetaT],
        dst: TorchaoGpu,
        *,
        non_blocking: bool = False,
    ) -> None:
        copy_storage(src.storage, dst.storage, non_blocking=non_blocking)

    @classmethod
    def cache_bytes(cls, state: TorchaoPinned[MetaT]) -> int:
        return sum(s.nbytes for s in state.storage if s is not None)

    @classmethod
    def compute_dtype(cls, t: torch.Tensor) -> torch.dtype:
        # Validate once through the shared _require, then read the logical
        # dtype off the wrapper — same single-validation path as the other
        # hooks (subclasses implement _compute_dtype on the wrapper).
        return cls._compute_dtype(cls._require(t))
