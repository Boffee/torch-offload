"""Shared pinned binding records and slot movement helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .pinned_buffer import PinnedBuffer
from .pinned_param import PinnedParam
from .slots import (
    BufferSlot,
    ModuleSlotCollection,
    ParamSlot,
    set_param_data,
    unique_slots,
)

PostCopyHook = Callable[[nn.Parameter], None]


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


@dataclass(slots=True)
class PinnedParamBinding:
    """One model instance's parameter slots bound to one pinned backing."""

    name: str
    pinned: PinnedParam
    slots: list[ParamSlot]
    cpu_param: nn.Parameter

    @property
    def requires_grad(self) -> bool:
        return self.pinned.requires_grad

    @property
    def cache_bytes(self) -> int:
        return self.pinned.cache_bytes

    @property
    def unique_slots(self) -> list[ParamSlot]:
        return unique_slots(self.slots)

    def allocate_gpu_storage(self, device: torch.device) -> object:
        return self.pinned.allocate_gpu_storage(device)

    def make_gpu_param(self, target_state: object) -> nn.Parameter:
        return self.pinned.make_gpu_param(target_state)

    def copy_to_gpu(
        self,
        target_state: object,
        *,
        non_blocking: bool = False,
    ) -> None:
        self.pinned.copy_to_gpu(target_state, non_blocking=non_blocking)

    def validate_parameter_data_swap_target(self) -> None:
        self.pinned.validate_parameter_data_swap_target()

    def set_slots(self, param: nn.Parameter) -> None:
        """Set every managed parameter slot to ``param``.

        Trainable params keep the user's ``Parameter`` object and swap
        only ``.data`` so autograd hooks and optimizer references stay
        attached to the same wrapper.
        """
        if self.requires_grad:
            for slot in self.unique_slots:
                set_param_data(slot.get(), param.data)
        else:
            for slot in self.unique_slots:
                slot.set(param)

    def restore_pinned(self) -> None:
        """Repoint managed parameter slots to their pinned CPU wrapper."""
        self.set_slots(self.cpu_param)


@dataclass(slots=True)
class PinnedBufferBinding:
    """One model instance's buffer slots bound to one pinned backing.

    Active-time buffer mutations are discarded on restore:
    :meth:`restore_pinned` reinstalls this pinned clone rather than
    copying active buffer state back.
    """

    name: str
    pinned: PinnedBuffer
    slots: list[BufferSlot]

    @property
    def cache_bytes(self) -> int:
        return self.pinned.cache_bytes

    @property
    def unique_slots(self) -> list[BufferSlot]:
        return unique_slots(self.slots)

    def allocate_gpu_buffer(self, device: torch.device) -> torch.Tensor:
        return torch.empty_like(self.pinned.tensor, device=device)

    def copy_to_gpu(
        self,
        target: torch.Tensor,
        *,
        non_blocking: bool = False,
    ) -> None:
        target.copy_(self.pinned.tensor, non_blocking=non_blocking)

    def set_slots(self, buffer: torch.Tensor) -> None:
        """Set every managed buffer slot to ``buffer``."""
        for slot in self.unique_slots:
            slot.set(buffer)

    def restore_pinned(self) -> None:
        """Repoint managed buffer slots to the pinned CPU clone."""
        self.set_slots(self.pinned.tensor)


class PinnedModuleTarget:
    """GPU storage target for one :class:`PinnedModuleBinding` layout.

    The target owns adapter-specific GPU storage for every pinned param
    and reusable GPU tensors for every pinned buffer in a
    binding-compatible module scope. Repeated load calls copy pinned CPU
    bytes into the same GPU storage and return the same active object
    for a given pinned state name.

    Binding names are target storage
    keys. They are intentionally the module-relative PyTorch names:
    whole-model targets use fully qualified names, while streamed block
    targets use block-local names so every block can load into the same
    pool layout. Names must be unique within each target namespace.

    Construction snapshots the binding's pinned layout. The target does
    not retain the template binding as the source of load bytes; any
    compatible binding can later load into the target.
    """

    __slots__ = (
        "_device",
        "_target_buffers",
        "_target_params",
        "_target_states",
    )

    def __init__(
        self,
        binding: PinnedModuleBinding,
        *,
        device: torch.device,
    ) -> None:
        if device.type != "cuda":
            raise ValueError(
                "PinnedModuleTarget requires a CUDA device; "
                f"got {device}."
            )

        # Validate the whole target layout before allocating any GPU storage.
        seen_param_names: set[str] = set()
        for param_binding in binding.param_bindings:
            if param_binding.name in seen_param_names:
                raise ValueError(
                    "PinnedModuleTarget requires unique pinned param target "
                    f"names; got duplicate {param_binding.name!r}."
                )
            seen_param_names.add(param_binding.name)

        seen_buffer_names: set[str] = set()
        for buffer_binding in binding.buffer_bindings:
            if buffer_binding.name in seen_buffer_names:
                raise ValueError(
                    "PinnedModuleTarget requires unique pinned buffer target "
                    f"names; got duplicate {buffer_binding.name!r}."
                )
            seen_buffer_names.add(buffer_binding.name)

        self._device = device

        # Allocate reusable active storage for the validated layout.
        self._target_states: dict[str, object] = {}
        self._target_params: dict[str, nn.Parameter] = {}
        for param_binding in binding.param_bindings:
            target_state = param_binding.allocate_gpu_storage(device)
            self._target_states[param_binding.name] = target_state
            self._target_params[param_binding.name] = param_binding.make_gpu_param(
                target_state
            )
        self._target_buffers: dict[str, torch.Tensor] = {}
        for buffer_binding in binding.buffer_bindings:
            self._target_buffers[buffer_binding.name] = (
                buffer_binding.allocate_gpu_buffer(device)
            )

    @property
    def device(self) -> torch.device:
        return self._device

    def load_param_bindings(
        self,
        param_bindings: Sequence[PinnedParamBinding],
        *,
        non_blocking: bool = False,
    ) -> dict[str, nn.Parameter]:
        """Copy ``param_bindings`` into this target.

        Returns the stable GPU ``Parameter`` wrappers keyed by pinned
        target name.
        """
        target_params: dict[str, nn.Parameter] = {}
        for binding in param_bindings:
            binding.copy_to_gpu(
                self._target_states[binding.name],
                non_blocking=non_blocking,
            )
            target_params[binding.name] = self._target_params[binding.name]
        return target_params

    def load_buffer_bindings(
        self,
        buffer_bindings: Sequence[PinnedBufferBinding],
        *,
        non_blocking: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Copy ``buffer_bindings`` into this target.

        Returns the stable GPU tensors keyed by pinned target name.
        """
        target_buffers: dict[str, torch.Tensor] = {}
        for binding in buffer_bindings:
            target = self._target_buffers[binding.name]
            binding.copy_to_gpu(target, non_blocking=non_blocking)
            target_buffers[binding.name] = target
        return target_buffers


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
            total += param_binding.cache_bytes
        for buffer_binding in self.buffer_bindings:
            total += buffer_binding.cache_bytes
        return total

    def contains_param_binding(self, binding: PinnedParamBinding) -> bool:
        return any(binding is candidate for candidate in self.param_bindings)

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
        """Copy pinned storage into ``target`` and set managed slots.

        Slot mutation is deliberately delayed until after all target
        copies and post-copy hooks complete, so a copy failure does not
        leave the module partially active.
        """
        target_params = target.load_param_bindings(
            self.param_bindings, non_blocking=non_blocking,
        )
        for param_binding in self.param_bindings:
            target_param = target_params[param_binding.name]
            hook = (
                post_copy_hooks.get(id(param_binding))
                if post_copy_hooks is not None
                else None
            )
            if hook is not None:
                hook(target_param)

        target_buffers = target.load_buffer_bindings(
            self.buffer_bindings, non_blocking=non_blocking,
        )

        for param_binding in self.param_bindings:
            param_binding.set_slots(target_params[param_binding.name])

        for buffer_binding in self.buffer_bindings:
            buffer_binding.set_slots(target_buffers[buffer_binding.name])

    def iter_trainables(
        self,
    ) -> Iterator[tuple[PinnedParamBinding, nn.Module, str]]:
        for param_binding in self.param_bindings:
            if param_binding.requires_grad:
                slot = param_binding.unique_slots[0]
                yield param_binding, slot.parent, slot.leaf

    def has_trainables(self) -> bool:
        return any(param_binding.requires_grad for param_binding in self.param_bindings)


ParamPinValidator = Callable[[PinnedParamBinding], None]


def _bind_param_slots(
    name: str, pinned: PinnedParam, slots: Sequence[ParamSlot],
) -> PinnedParamBinding:
    """Bind live model slots to an existing pinned parameter backing."""
    slot_list = list(slots)
    if not slot_list:
        raise ValueError("_bind_param_slots requires at least one ParamSlot")
    return PinnedParamBinding(
        name=name,
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
    name = primary_slot.name
    pinned = PinnedParam(name, primary_slot.get())
    binding = _bind_param_slots(name, pinned, slot_list)
    if validate_param is not None:
        validate_param(binding)
    return binding


def _pin_buffer_slots(slots: Sequence[BufferSlot]) -> PinnedBufferBinding:
    """Clone and pin the first slot's buffer and bind all aliases to it."""
    slot_list = list(slots)
    if not slot_list:
        raise ValueError("_pin_buffer_slots requires at least one BufferSlot")
    primary_slot = slot_list[0]
    name = primary_slot.name
    pinned = PinnedBuffer.clone(name, primary_slot.get())
    return PinnedBufferBinding(name=name, pinned=pinned, slots=slot_list)


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
    "PostCopyHook",
    "PostCopyHookHandle",
    "pin_module_slot_collection",
]
