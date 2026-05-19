"""Shared pinned binding records and slot movement helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .pinned_param import PinnedParam, PostCopyHook
from .slots import (
    BufferSlot,
    ModuleSlotCollection,
    ParamSlot,
    set_param_data,
    unique_slots,
)
from .tensor_adapters import clone_to_pinned_cpu


@dataclass(slots=True)
class PinnedParamBinding:
    """One model instance's parameter slots bound to one pinned backing."""

    pinned: PinnedParam
    slots: list[ParamSlot]
    cpu_param: nn.Parameter

    @property
    def unique_slots(self) -> list[ParamSlot]:
        return unique_slots(self.slots)


@dataclass(slots=True)
class PinnedBufferBinding:
    """One model instance's buffer slots bound to one pinned tensor."""

    pinned: torch.Tensor
    slots: list[BufferSlot]

    @property
    def unique_slots(self) -> list[BufferSlot]:
        return unique_slots(self.slots)


class PinnedModuleTarget:
    """GPU storage target for one :class:`PinnedModuleBinding` layout.

    The target owns adapter-specific GPU storage for every pinned param
    in a binding-compatible module scope. Repeated :meth:`load_param`
    calls copy pinned CPU bytes into the same GPU storage and return the
    same ``nn.Parameter`` wrapper for a given param name.
    """

    __slots__ = ("_device", "_gpu_params", "_gpu_states")

    def __init__(
        self, pinned_params: Sequence[PinnedParam], device: torch.device,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(
                "PinnedModuleTarget requires a CUDA device; "
                f"got {device}."
            )
        self._device = device
        self._gpu_states: dict[str, object] = {}
        self._gpu_params: dict[str, nn.Parameter] = {}
        for pinned in pinned_params:
            gpu_state = pinned.allocate_gpu_storage(device)
            self._gpu_states[pinned.name] = gpu_state
            self._gpu_params[pinned.name] = pinned.make_gpu_param(gpu_state)

    @property
    def device(self) -> torch.device:
        return self._device

    def load_param(
        self,
        pinned: PinnedParam,
        *,
        non_blocking: bool = False,
    ) -> nn.Parameter:
        """Copy ``pinned`` into this target and return its GPU Parameter."""
        pinned.copy_to_gpu(
            self._gpu_states[pinned.name], non_blocking=non_blocking,
        )
        return self._gpu_params[pinned.name]


@dataclass(slots=True)
class PinnedModuleBinding:
    """Pinned storage bound to slots collected from one module scope.

    The binding does not own the ``nn.Module``. It owns the pinned
    param/buffer bindings and knows how to install either pinned CPU
    storage or GPU materializations back into those slots.
    """

    param_bindings: list[PinnedParamBinding]
    buffer_bindings: list[PinnedBufferBinding]

    @property
    def cache_bytes(self) -> int:
        total = 0
        for param_binding in self.param_bindings:
            total += param_binding.pinned.cache_bytes
        for buffer_binding in self.buffer_bindings:
            total += (
                buffer_binding.pinned.numel()
                * buffer_binding.pinned.element_size()
            )
        return total

    @property
    def pinned_params(self) -> list[PinnedParam]:
        return [
            param_binding.pinned for param_binding in self.param_bindings
        ]

    def contains_pinned_param(self, pinned: PinnedParam) -> bool:
        return any(
            param_binding.pinned is pinned
            for param_binding in self.param_bindings
        )

    def place_on_pinned(self) -> None:
        """Restore managed slots to their pinned CPU forms.

        Frozen params use slot replacement. Trainable params keep the
        user's ``Parameter`` object and repoint only ``.data``.
        """
        for param_binding in self.param_bindings:
            if param_binding.pinned.requires_grad:
                for slot in param_binding.unique_slots:
                    set_param_data(
                        slot.get(), param_binding.cpu_param.data
                    )
            else:
                for slot in param_binding.unique_slots:
                    slot.set(param_binding.cpu_param)
        for buffer_binding in self.buffer_bindings:
            for slot in buffer_binding.unique_slots:
                slot.set(buffer_binding.pinned)

    def place_on_gpu(
        self,
        target: PinnedModuleTarget,
        *,
        post_copy_hooks: Mapping[int, PostCopyHook] | None = None,
        non_blocking: bool = False,
    ) -> None:
        """Copy pinned storage into ``target`` and install GPU slots."""
        gpu_params: list[tuple[PinnedParamBinding, nn.Parameter]] = []
        for param_binding in self.param_bindings:
            gpu_param = target.load_param(
                param_binding.pinned, non_blocking=non_blocking,
            )
            hook = (
                post_copy_hooks.get(id(param_binding.pinned))
                if post_copy_hooks is not None
                else None
            )
            if hook is not None:
                hook(gpu_param)
            gpu_params.append((param_binding, gpu_param))

        for param_binding, gpu_param in gpu_params:
            if param_binding.pinned.requires_grad:
                for slot in param_binding.unique_slots:
                    set_param_data(slot.get(), gpu_param.data)
            else:
                for slot in param_binding.unique_slots:
                    slot.set(gpu_param)

        for buffer_binding in self.buffer_bindings:
            gpu = buffer_binding.pinned.to(
                target.device, non_blocking=non_blocking,
            )
            for slot in buffer_binding.unique_slots:
                slot.set(gpu)

    def iter_trainables(
        self,
    ) -> Iterator[tuple[PinnedParamBinding, nn.Module, str]]:
        for param_binding in self.param_bindings:
            if param_binding.pinned.requires_grad:
                slot = param_binding.unique_slots[0]
                yield param_binding, slot.parent, slot.leaf

    def has_trainables(self) -> bool:
        return any(
            param_binding.pinned.requires_grad
            for param_binding in self.param_bindings
        )


ParamPinValidator = Callable[[PinnedParam, ParamSlot], None]


def _bind_param_slots(
    pinned: PinnedParam, slots: Sequence[ParamSlot],
) -> PinnedParamBinding:
    """Bind live model slots to an existing pinned parameter backing."""
    slot_list = list(slots)
    if not slot_list:
        raise ValueError("_bind_param_slots requires at least one ParamSlot")
    return PinnedParamBinding(
        pinned=pinned,
        slots=slot_list,
        cpu_param=pinned.make_cpu_param(),
    )


def _pin_param_slots(
    slots: Sequence[ParamSlot],
    *,
    validate_param: ParamPinValidator | None = None,
) -> PinnedParamBinding:
    """Pin the first slot's parameter and bind all aliases to that backing."""
    slot_list = list(slots)
    if not slot_list:
        raise ValueError("_pin_param_slots requires at least one ParamSlot")
    primary_slot = slot_list[0]
    pinned = PinnedParam(primary_slot.name, primary_slot.get())
    if validate_param is not None:
        validate_param(pinned, primary_slot)
    return _bind_param_slots(pinned, slot_list)


def _pin_buffer_slots(slots: Sequence[BufferSlot]) -> PinnedBufferBinding:
    """Clone and pin the first slot's buffer and bind all aliases to it."""
    slot_list = list(slots)
    if not slot_list:
        raise ValueError("_pin_buffer_slots requires at least one BufferSlot")
    pinned = clone_to_pinned_cpu(
        slot_list[0].get(),
        memory_format=torch.contiguous_format,
    )
    return PinnedBufferBinding(pinned=pinned, slots=slot_list)


def _pin_module_slots(
    param_slot_groups: Sequence[Sequence[ParamSlot]],
    buffer_slot_groups: Sequence[Sequence[BufferSlot]],
    *,
    validate_param: ParamPinValidator | None = None,
) -> PinnedModuleBinding:
    """Pin grouped module slots into one aggregate binding."""
    return PinnedModuleBinding(
        param_bindings=[
            _pin_param_slots(slots, validate_param=validate_param)
            for slots in param_slot_groups
        ],
        buffer_bindings=[
            _pin_buffer_slots(slots) for slots in buffer_slot_groups
        ],
    )


def pin_module_slot_collection(
    collection: ModuleSlotCollection,
    *,
    validate_param: ParamPinValidator | None = None,
) -> PinnedModuleBinding:
    """Pin collected module slots into one aggregate binding."""
    return _pin_module_slots(
        collection.param_slot_groups,
        collection.buffer_slot_groups,
        validate_param=validate_param,
    )


__all__ = [
    "PinnedBufferBinding",
    "PinnedModuleBinding",
    "PinnedModuleTarget",
    "PinnedParamBinding",
    "pin_module_slot_collection",
]
