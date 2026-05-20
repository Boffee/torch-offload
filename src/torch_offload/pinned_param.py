"""Per-parameter pinned-CPU storage primitive.

Internal to the ``torch_offload`` subpackage. Shared by
:class:`PinnedWeights` (whole-model bulk pin) and :class:`StreamedWeights`
(per-block streaming). Both consumers reach this through the same
abstraction so the addition of new tensor types only requires writing
a new :class:`TensorAdapter`, not editing the consumers.

Per-parameter mechanics live in the tensor adapter
(:mod:`tensor_adapters` for plain tensors, :mod:`quanto_adapter` for
quanto). :class:`PinnedParam` is a thin holder that pairs one
:class:`nn.Parameter` with the adapter that handles its tensor type
plus the pinned-host state that adapter produced. Optional operations
such as D2H round-trip, trainable ``.data`` swap, and dense updates
are exposed through adapter capability methods.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .slots import set_param_data
from .tensor_adapter_factory import select_adapter
from .tensor_adapter_factory import storage_key as _storage_key
from .tensor_adapters import (
    CpuRoundTripTensorAdapter,
    ParameterDataSwapTensorAdapter,
    TensorAdapter,
    adapter_name,
)


def storage_key(t: torch.Tensor) -> tuple[Any, ...]:
    """Compatibility wrapper for :func:`tensor_adapter_factory.storage_key`."""
    return _storage_key(t)


class PinnedParam:
    """Pinned host storage for one parameter, with GPU-load helpers.

    Construction picks an adapter via :func:`select_adapter` based on
    the parameter's tensor type, then uses the adapter to clone-and-pin
    the bytes. Model-bound callers create their own
    deactivated-state :class:`nn.Parameter` wrappers with
    :meth:`make_cpu_param`.

    The lifecycle methods (:meth:`make_cpu_param`,
    :meth:`allocate_gpu_storage`, :meth:`make_gpu_param`,
    :meth:`copy_to_gpu`, :meth:`copy_to_cpu`) all dispatch through the
    adapter. This primitive works with the opaque state returned by
    :meth:`allocate_gpu_storage`; model-bound callers should normally
    use :class:`PinnedParamBinding`, which wraps that state in a
    binding-owned target object.

    The pinned parameter captures the source parameter's ``requires_grad`` at
    construction time and threads it through to the adapter when
    building CPU and GPU parameter wrappers. Frozen
    callers (:class:`PinnedWeights`, ``PinnedModuleBinding`` for
    ``requires_grad=False`` slots) get the historic behavior. Trainable
    callers can either request the wrapper preserve ``requires_grad``
    or skip the wrapper entirely and ``.data``-swap into their own
    persistent Parameter — both are supported.

    Low-peak construction behavior: for plain ``torch.Tensor`` parameters,
    construction immediately repoints the source ``Parameter.data`` at the
    pinned clone. This releases the original source storage before the
    owning strategy finishes constructing every pinned parameter, avoiding a temporary
    2x peak for large CPU-resident models and promptly freeing GPU storage
    for CUDA-origin models. It also means construction is not
    rollback-safe after pinning has started: if a later pinned parameter fails
    to pin, recovery of the partially constructed strategy/model is unsupported.
    Drop those references and rebuild from a fresh model instance. Tensor
    subclasses skip this optimization because ``.data =`` can drop wrapper
    state.
    """

    __slots__ = ("_target_layout", "adapter", "pinned_state", "requires_grad")

    def __init__(self, param: nn.Parameter) -> None:
        self.adapter: TensorAdapter[Any, Any] = select_adapter(param.data)
        self._target_layout = self._target_layout_from_adapter(
            self.adapter, param.data,
        )
        self.requires_grad: bool = param.requires_grad
        self.pinned_state = self.adapter.clone_pin(param.data)
        # Low-peak construction optimization: release the original source
        # storage by repointing the source Parameter at the
        # pinned clone immediately. This is an intentional mutation of
        # the caller's model before the owning strategy has finished
        # construction; see the class docstring for failure semantics.
        # Only safe for plain tensors; subclass wrappers can lose
        # metadata or ignore .data assignment.
        if type(param.data) is torch.Tensor:
            set_param_data(param, self.make_cpu_param().data)

    @staticmethod
    def _target_layout_from_adapter(
        adapter: TensorAdapter[Any, Any], tensor: torch.Tensor,
    ) -> tuple[object, object]:
        return (type(adapter), adapter.layout_signature(tensor))

    @classmethod
    def target_layout_for(cls, param: nn.Parameter) -> tuple[object, object]:
        """Opaque target-compatibility layout for ``param``.

        Used by pre-pin validation paths that must fail before
        :class:`PinnedParam` mutates the source parameter. Callers should
        compare the returned value for equality only.
        """
        adapter = select_adapter(param.data)
        return cls._target_layout_from_adapter(adapter, param.data)

    @property
    def target_layout(self) -> tuple[object, object]:
        """Opaque target-compatibility layout for this pinned backing."""
        return self._target_layout

    def make_cpu_param(self) -> nn.Parameter:
        """Build a CPU :class:`nn.Parameter` wrapper over this pinned state.

        Each model-bound binding owns its own wrapper object, while the
        underlying pinned storage remains shared through this
        :class:`PinnedParam`.
        """
        return self.adapter.cpu_param(
            self.pinned_state, requires_grad=self.requires_grad,
        )

    def allocate_gpu_storage(self, device: torch.device) -> object:
        """Allocate empty GPU storage mirroring this pinned parameter's layout.
        Returns an opaque adapter-specific handle; pass it back to
        :meth:`make_gpu_param`, :meth:`copy_to_gpu`, and
        :meth:`copy_to_cpu`."""
        return self.adapter.alloc_gpu(self.pinned_state, device)

    def make_gpu_param(self, gpu_state: object) -> nn.Parameter:
        """Build the GPU-side :class:`nn.Parameter` for this pinned parameter.
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
                f"{adapter_name(self.adapter)} does not support CPU round-trip: "
                "its GPU representation cannot be copied back into the "
                "pinned host state without adapter-specific conversion."
            )
        self.adapter.copy_to_cpu(
            gpu_state, self.pinned_state, non_blocking=non_blocking
        )

    @property
    def compute_dtype(self) -> torch.dtype:
        """Logical compute dtype reported by this pinned parameter's adapter."""
        return self.adapter.compute_dtype(self.make_cpu_param().data)

    def validate_parameter_data_swap_target(self) -> None:
        """Raise if this pinned parameter cannot use trainable streaming.

        CUDA trainable streaming requires both Parameter.data swap
        compatibility and a D2H path back into pinned host state after
        optimizer updates.
        """
        if not isinstance(self.adapter, ParameterDataSwapTensorAdapter):
            raise NotImplementedError(
                f"Trainable streaming requires a Parameter.data-swap-capable "
                f"tensor adapter; this parameter uses {adapter_name(self.adapter)}. "
                "Quantized or structured weights are inference-only here — "
                "keep them frozen, or wrap with PEFT/LoRA so the trainable "
                "adapter weights are plain tensors."
            )
        try:
            self.adapter.validate_parameter_data_swap_target(
                self.make_cpu_param().data
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"Trainable streaming cannot use Parameter.data swap: {exc}"
            ) from exc
        if not isinstance(self.adapter, CpuRoundTripTensorAdapter):
            raise NotImplementedError(
                f"Trainable streaming requires a CPU-round-trip-capable "
                f"tensor adapter; this parameter uses {adapter_name(self.adapter)}."
            )

    @property
    def cache_bytes(self) -> int:
        """Bytes this pinned parameter consumes in pinned host memory."""
        return self.adapter.cache_bytes(self.pinned_state)
