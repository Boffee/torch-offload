"""Shared pinned binding records and slot movement helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .pinned_buffer import PinnedBuffer
from .pinned_param import PinnedParam, PostCopyHook
from .slots import (
    BufferSlot,
    ModuleSlotCollection,
    ParamSlot,
    set_param_data,
    unique_slots,
)


@dataclass(slots=True)
class PinnedParamBinding:
    """One model instance's parameter slots bound to one pinned backing."""

    pinned: PinnedParam
    slots: list[ParamSlot]
    cpu_param: nn.Parameter

    @property
    def unique_slots(self) -> list[ParamSlot]:
        return unique_slots(self.slots)

    def set_slots(self, param: nn.Parameter) -> None:
        """Set every managed parameter slot to ``param``.

        Trainable params keep the user's ``Parameter`` object and swap
        only ``.data`` so autograd hooks and optimizer references stay
        attached to the same wrapper.
        """
        if self.pinned.requires_grad:
            for slot in self.unique_slots:
                set_param_data(slot.get(), param.data)
        else:
            for slot in self.unique_slots:
                slot.set(param)

    def restore_pinned(self) -> None:
        """Repoint managed parameter slots to their pinned CPU wrapper."""
        self.set_slots(self.cpu_param)

    def copy_to_target(
        self,
        target: PinnedModuleTarget,
        *,
        post_copy_hooks: Mapping[int, PostCopyHook] | None = None,
        non_blocking: bool = False,
    ) -> nn.Parameter:
        """Copy pinned storage into ``target`` and return the target wrapper."""
        target_param = target.load_param(
            self.pinned, non_blocking=non_blocking,
        )
        hook = (
            post_copy_hooks.get(id(self.pinned))
            if post_copy_hooks is not None
            else None
        )
        if hook is not None:
            hook(target_param)
        return target_param

    def load_to_target(
        self,
        target: PinnedModuleTarget,
        *,
        post_copy_hooks: Mapping[int, PostCopyHook] | None = None,
        non_blocking: bool = False,
    ) -> None:
        """Copy pinned storage into ``target`` and set managed slots."""
        param = self.copy_to_target(
            target,
            post_copy_hooks=post_copy_hooks,
            non_blocking=non_blocking,
        )
        self.set_slots(param)


@dataclass(slots=True)
class PinnedBufferBinding:
    """One model instance's buffer slots bound to one pinned backing.

    Active-time buffer mutations are discarded on restore:
    :meth:`restore_pinned` reinstalls this pinned clone rather than
    copying active buffer state back.
    """

    pinned: PinnedBuffer
    slots: list[BufferSlot]

    @property
    def unique_slots(self) -> list[BufferSlot]:
        return unique_slots(self.slots)

    def set_slots(self, buffer: torch.Tensor) -> None:
        """Set every managed buffer slot to ``buffer``."""
        for slot in self.unique_slots:
            slot.set(buffer)

    def restore_pinned(self) -> None:
        """Repoint managed buffer slots to the pinned CPU clone."""
        self.set_slots(self.pinned.tensor)

    def copy_to_target(
        self,
        target: PinnedModuleTarget,
        *,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        """Copy the pinned clone to ``target`` and return the target tensor."""
        return self.pinned.tensor.to(
            target.device, non_blocking=non_blocking,
        )

    def load_to_target(
        self,
        target: PinnedModuleTarget,
        *,
        non_blocking: bool = False,
    ) -> None:
        """Copy the pinned clone to ``target`` and set managed slots."""
        buffer = self.copy_to_target(target, non_blocking=non_blocking)
        self.set_slots(buffer)


class PinnedModuleTarget:
    """GPU storage target for one :class:`PinnedModuleBinding` layout.

    The target owns adapter-specific GPU storage for every pinned param
    in a binding-compatible module scope. Repeated :meth:`load_param`
    calls copy pinned CPU bytes into the same GPU storage and return the
    same ``nn.Parameter`` wrapper for a given pinned param name.

    ``PinnedParam.name`` is the target storage key. It is intentionally
    the module-relative PyTorch parameter name: whole-model targets use
    fully qualified names, while streamed block targets use block-local
    names so every block can load into the same pool layout. Names must
    be unique within one target.
    """

    __slots__ = ("_device", "_target_params", "_target_states")

    def __init__(
        self, pinned_params: Sequence[PinnedParam], device: torch.device,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(
                "PinnedModuleTarget requires a CUDA device; "
                f"got {device}."
            )
        self._device = device
        seen_names: set[str] = set()
        for pinned in pinned_params:
            if pinned.name in seen_names:
                raise ValueError(
                    "PinnedModuleTarget requires unique pinned target "
                    f"names; got duplicate {pinned.name!r}."
                )
            seen_names.add(pinned.name)

        self._target_states: dict[str, object] = {}
        self._target_params: dict[str, nn.Parameter] = {}
        for pinned in pinned_params:
            target_state = pinned.allocate_gpu_storage(device)
            self._target_states[pinned.name] = target_state
            self._target_params[pinned.name] = pinned.make_gpu_param(
                target_state
            )

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
            self._target_states[pinned.name], non_blocking=non_blocking,
        )
        return self._target_params[pinned.name]


@dataclass(slots=True)
class PinnedModuleBinding:
    """Pinned storage bound to slots collected from one module scope.

    The binding does not own the ``nn.Module``. It owns the pinned
    param/buffer bindings and knows how to set either pinned CPU
    storage or target materializations back into those slots.
    """

    param_bindings: list[PinnedParamBinding]
    buffer_bindings: list[PinnedBufferBinding]

    @property
    def cache_bytes(self) -> int:
        total = 0
        for param_binding in self.param_bindings:
            total += param_binding.pinned.cache_bytes
        for buffer_binding in self.buffer_bindings:
            total += buffer_binding.pinned.cache_bytes
        return total

    @property
    def pinned_params(self) -> list[PinnedParam]:
        return [
            param_binding.pinned for param_binding in self.param_bindings
        ]

    @property
    def pinned_buffers(self) -> list[PinnedBuffer]:
        return [
            buffer_binding.pinned for buffer_binding in self.buffer_bindings
        ]

    def contains_pinned_param(self, pinned: PinnedParam) -> bool:
        return any(
            param_binding.pinned is pinned
            for param_binding in self.param_bindings
        )

    def restore_pinned(self) -> None:
        """Restore managed slots to their pinned CPU forms.

        Frozen params use slot replacement. Trainable params keep the
        user's ``Parameter`` object and repoint only ``.data``.
        """
        for param_binding in self.param_bindings:
            param_binding.restore_pinned()
        for buffer_binding in self.buffer_bindings:
            buffer_binding.restore_pinned()

    def load_to_target(
        self,
        target: PinnedModuleTarget,
        *,
        post_copy_hooks: Mapping[int, PostCopyHook] | None = None,
        non_blocking: bool = False,
    ) -> None:
        """Copy pinned storage into ``target`` and set managed slots."""
        target_params: list[tuple[PinnedParamBinding, nn.Parameter]] = []
        for param_binding in self.param_bindings:
            target_param = param_binding.copy_to_target(
                target,
                post_copy_hooks=post_copy_hooks,
                non_blocking=non_blocking,
            )
            target_params.append((param_binding, target_param))

        for param_binding, target_param in target_params:
            param_binding.set_slots(target_param)

        for buffer_binding in self.buffer_bindings:
            buffer_binding.load_to_target(
                target, non_blocking=non_blocking,
            )

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
    primary_slot = slot_list[0]
    pinned = PinnedBuffer.clone(primary_slot.name, primary_slot.get())
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
    "PinnedBuffer",
    "PinnedBufferBinding",
    "PinnedModuleBinding",
    "PinnedModuleTarget",
    "PinnedParamBinding",
    "pin_module_slot_collection",
]
