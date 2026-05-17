"""Per-parameter pinned-CPU storage primitive.

Internal to the ``torch_offload`` subpackage. Shared by
:class:`PinnedWeights` (whole-model bulk pin) and :class:`StreamedWeights`
(per-block streaming). Both consumers reach this through the same
abstraction so the addition of new tensor types only requires writing
a new :class:`TensorAdapter`, not editing the consumers.

Per-parameter mechanics live in the tensor adapter
(:mod:`tensor_adapters` for plain tensors, :mod:`quanto_adapter` for
quanto). :class:`PinnedParamBuffer` is a thin holder that pairs one
:class:`nn.Parameter` with the adapter that handles its tensor type
plus the pinned-host state that adapter produced. Optional operations
such as D2H round-trip, trainable ``.data`` swap, and dense updates
are exposed through adapter capability methods.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

import torch
from torch import nn

from . import (
    gguf_adapter,  # noqa: F401 — registration side effect
    nvfp4_adapter,  # noqa: F401 — registration side effect
    quanto_adapter,  # noqa: F401 — registration side effect
)
from .tensor_adapters import (
    CpuRoundTripTensorAdapter,
    ParameterDataSwapTensorAdapter,
    TensorAdapter,
    select_adapter,
)

PostCopyHook = Callable[[torch.Tensor], None]


class PostCopyHookHandle:
    """Removal handle returned by post-copy hook registration."""

    __slots__ = ("_hooks", "_key")

    def __init__(
        self, hooks: MutableMapping[int, PostCopyHook], key: int,
    ) -> None:
        self._hooks: MutableMapping[int, PostCopyHook] | None = hooks
        self._key = key

    def remove(self) -> None:
        hooks = self._hooks
        if hooks is None:
            return
        hooks.pop(self._key, None)
        self._hooks = None


def storage_key(t: torch.Tensor) -> tuple[Any, ...]:
    """Identity key for tied-weight detection.

    Two tensors that produce the same key represent the same logical
    tensor backed by the same storage region with the same view layout
    and (for quanto) the same quant metadata; they can be deduplicated
    into a single :class:`PinnedParamBuffer`.

    Used by :class:`~torch_offload.PinnedWeights` (for handle-level
    dedup of tied frozen params) and
    :func:`~torch_offload.ModelOffloader` (for cross-region
    tied-weight detection across blocks and non-block modules).

    Dispatches to the matching adapter so each tensor type contributes
    its own identity components (regular: storage + view; quanto:
    storage of both inner tensors plus quant metadata).
    """
    return select_adapter(t).storage_key(t)


class PinnedParamBuffer:
    """Pinned host storage for one parameter, with GPU-load helpers.

    Construction picks an adapter via :func:`select_adapter` based on
    the parameter's tensor type, then uses the adapter to clone-and-pin
    the bytes and build the deactivated-state :class:`nn.Parameter`
    (:attr:`cpu_param`).

    The lifecycle methods (:meth:`allocate_gpu_storage`,
    :meth:`make_gpu_param`, :meth:`copy_to_gpu`, :meth:`load_to_gpu`)
    all dispatch through the adapter. Consumers work with the opaque
    :class:`GpuState` returned by :meth:`allocate_gpu_storage`; the
    buffer round-trips that opaque handle through subsequent calls.

    The buffer captures the source parameter's ``requires_grad`` at
    construction time and threads it through to the adapter when
    building :attr:`cpu_param` and the pool's ``gpu_param``. Frozen
    callers (:class:`PinnedWeights`, ``_BlockPinnedStore`` for
    ``requires_grad=False`` slots) get the historic behavior. Trainable
    callers can either request the wrapper preserve ``requires_grad``
    or skip the wrapper entirely and ``.data``-swap into their own
    persistent Parameter — both are supported.

    Low-peak construction behavior: for plain ``torch.Tensor`` parameters,
    construction immediately repoints the source ``Parameter.data`` at the
    pinned clone. This releases the original pageable CPU storage before the
    owning strategy finishes constructing every buffer, avoiding a temporary
    2x host-memory peak for large models. It also means construction is not
    rollback-safe after pinning has started: if a later buffer fails to pin,
    recovery of the partially constructed strategy/model is unsupported.
    Drop those references and rebuild from a fresh model instance. Tensor
    subclasses skip this optimization because ``.data =`` can drop wrapper
    state.
    """

    __slots__ = (
        "adapter", "cpu_param", "name", "pinned_state",
        "requires_grad",
    )

    def __init__(self, name: str, param: nn.Parameter) -> None:
        self.name = name
        self.adapter: type[TensorAdapter] = select_adapter(param.data)
        self.requires_grad: bool = param.requires_grad
        self.pinned_state = self.adapter.clone_pin(param.data)
        self.cpu_param: nn.Parameter = self.adapter.cpu_param(
            self.pinned_state, requires_grad=self.requires_grad,
        )
        # Low-peak construction optimization: release the original
        # pageable storage by repointing the source Parameter at the
        # pinned clone immediately. This is an intentional mutation of
        # the caller's model before the owning strategy has finished
        # construction; see the class docstring for failure semantics.
        # Only safe for plain tensors; subclass wrappers can lose
        # metadata or ignore .data assignment.
        if type(param.data) is torch.Tensor:
            param.data = self.cpu_param.data

    def allocate_gpu_storage(self, device: torch.device) -> object:
        """Allocate empty GPU storage mirroring this buffer's layout.
        Returns an opaque adapter-specific handle; pass it back to
        :meth:`make_gpu_param`, :meth:`copy_to_gpu`, and
        :meth:`copy_to_cpu`."""
        return self.adapter.alloc_gpu(self.pinned_state, device)

    def make_gpu_param(self, gpu_state: object) -> nn.Parameter:
        """Build the GPU-side :class:`nn.Parameter` for this buffer.
        Adapter receives the paired pinned state so structured tensor
        types (quanto) can reconstruct their wrappers. The wrapper's
        ``requires_grad`` matches the source parameter's at pin time."""
        return self.adapter.gpu_param(
            self.pinned_state, gpu_state, requires_grad=self.requires_grad,
        )

    def copy_to_gpu(self, gpu_state: object, *, non_blocking: bool = False) -> None:
        """Bulk DMA pinned host bytes into pre-allocated GPU storage."""
        self.adapter.copy_to_gpu(self.pinned_state, gpu_state, non_blocking=non_blocking)

    def copy_to_cpu(self, gpu_state: object, *, non_blocking: bool = False) -> None:
        """Bulk D2H GPU bytes back into the pinned host state.

        Optional symmetric counterpart to :meth:`copy_to_gpu`. The
        pinned state is overwritten in place with the current GPU
        contents — useful for syncing the host clone after an in-place
        GPU update (e.g., an optimizer step). Adapters whose GPU
        representation is not round-trippable do not expose this
        capability and raise :class:`NotImplementedError` here.
        """
        if not isinstance(self.adapter, CpuRoundTripTensorAdapter):
            raise NotImplementedError(
                f"{self.adapter.__name__} does not support CPU round-trip: "
                "its GPU representation cannot be copied back into the "
                "pinned host state without adapter-specific conversion."
            )
        self.adapter.copy_to_cpu(
            gpu_state, self.pinned_state, non_blocking=non_blocking
        )

    def load_to_gpu(
        self, device: torch.device, non_blocking: bool = False
    ) -> nn.Parameter:
        """Convenience: allocate GPU storage and copy in one shot.
        Used by :class:`PinnedWeights` (one-shot per-param load on
        activate); :class:`StreamedWeights` instead reuses a slot pool
        via :meth:`allocate_gpu_storage` + :meth:`make_gpu_param` once
        at pool construction and :meth:`copy_to_gpu` on each load."""
        gpu_state = self.allocate_gpu_storage(device)
        self.copy_to_gpu(gpu_state, non_blocking=non_blocking)
        return self.make_gpu_param(gpu_state)

    @property
    def compute_dtype(self) -> torch.dtype:
        """Logical compute dtype reported by this buffer's adapter."""
        return self.adapter.compute_dtype(self.cpu_param.data)

    def validate_parameter_data_swap_target(self, name: str) -> None:
        """Raise if this buffer cannot be trainable-streamed via ``.data``."""
        if not isinstance(self.adapter, ParameterDataSwapTensorAdapter):
            raise NotImplementedError(
                f"Trainable streaming requires a Parameter.data-swap-capable "
                f"tensor adapter; slot {name!r} uses {self.adapter.__name__}. "
                "Quantized or structured weights are inference-only here — "
                "keep them frozen, or wrap with PEFT/LoRA so the trainable "
                "adapter weights are plain tensors."
            )
        self.adapter.validate_parameter_data_swap_target(self.cpu_param.data, name)

    @property
    def cache_bytes(self) -> int:
        """Bytes this buffer consumes in pinned host memory."""
        return self.adapter.cache_bytes(self.pinned_state)
