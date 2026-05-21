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
class PinnedParamTarget:
    """Adapter-owned active storage for one pinned parameter binding.

    ``_state`` is intentionally private to the binding layer. It is
    the adapter-specific object returned by :class:`PinnedParam`; callers
    use ``param`` for slot mutation and pass the whole target back to the
    binding for H2D/D2H copies.
    """

    _state: object
    param: nn.Parameter


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
        """Pinned host bytes reachable through this parameter binding.

        Binding-level accounting is not an ownership claim: another
        binding may reference the same pinned backing.
        """
        return self.pinned.cache_bytes

    @property
    def target_layout(self) -> tuple[object, object]:
        return self.pinned.target_layout

    @property
    def unique_slots(self) -> list[ParamSlot]:
        return unique_slots(self.slots)

    def allocate_target(self, device: torch.device) -> PinnedParamTarget:
        """Allocate active storage for this binding on ``device``."""
        state = self.pinned.allocate_gpu_storage(device)
        return PinnedParamTarget(
            _state=state,
            param=self.pinned.make_gpu_param(state),
        )

    def copy_to_target(
        self,
        target: PinnedParamTarget,
        *,
        non_blocking: bool = False,
    ) -> None:
        """Copy pinned host bytes into ``target`` active storage."""
        self.pinned.copy_to_gpu(target._state, non_blocking=non_blocking)

    def copy_from_target(
        self,
        target: PinnedParamTarget,
        *,
        non_blocking: bool = False,
    ) -> None:
        """Copy active storage back into pinned host bytes.

        This is an explicit pinned-cache mutation path for trainable
        optimizer-step sync. Frozen inference code should not call it.
        """
        if not self.requires_grad:
            raise RuntimeError(
                "copy_from_target mutates pinned host backing and is only "
                f"valid for trainable param bindings; got {self.name!r}."
            )
        self.pinned.copy_to_cpu(target._state, non_blocking=non_blocking)

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
class PinnedBufferTarget:
    """Active tensor storage for one pinned buffer binding."""

    tensor: torch.Tensor


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
        """Pinned host bytes reachable through this buffer binding.

        Binding-level accounting is not an ownership claim: another
        binding may reference the same pinned backing.
        """
        return self.pinned.cache_bytes

    @property
    def target_layout(self) -> tuple[object, ...]:
        return self.pinned.target_layout

    @property
    def unique_slots(self) -> list[BufferSlot]:
        return unique_slots(self.slots)

    def allocate_target(self, device: torch.device) -> PinnedBufferTarget:
        """Allocate active storage for this buffer binding on ``device``."""
        return PinnedBufferTarget(
            tensor=torch.empty_like(self.pinned.tensor, device=device),
        )

    def copy_to_target(
        self,
        target: PinnedBufferTarget,
        *,
        non_blocking: bool = False,
    ) -> None:
        """Copy pinned host bytes into ``target`` active storage."""
        target.tensor.copy_(self.pinned.tensor, non_blocking=non_blocking)

    def set_slots(self, buffer: torch.Tensor) -> None:
        """Set every managed buffer slot to ``buffer``."""
        for slot in self.unique_slots:
            slot.set(buffer)

    def restore_pinned(self) -> None:
        """Repoint managed buffer slots to the pinned CPU clone."""
        self.set_slots(self.pinned.tensor)


def _validate_unique_binding_names(
    bindings: Sequence[PinnedParamBinding] | Sequence[PinnedBufferBinding],
    *,
    kind: str,
) -> None:
    seen_names: set[str] = set()
    for binding in bindings:
        if binding.name in seen_names:
            raise ValueError(
                f"PinnedModuleTarget requires unique pinned {kind} "
                f"target names; got duplicate {binding.name!r}."
            )
        seen_names.add(binding.name)


@dataclass(slots=True)
class PinnedModuleTarget:
    """GPU storage target for one module binding layout.

    The target owns active storage allocated by
    :meth:`PinnedModuleBinding.allocate_target`: adapter-specific GPU
    storage for every pinned param and reusable GPU tensors for every
    pinned buffer in a binding-compatible module scope.

    Binding names are target storage
    keys. They are intentionally the module-relative PyTorch names:
    whole-model targets use fully qualified names, while streamed block
    targets use block-local names so every block can load into the same
    pool layout. Names must be unique within each target namespace.

    The target does not retain the template binding as the source of load
    bytes; compatible bindings can later load into the target by name.
    """

    param_targets: dict[str, PinnedParamTarget]
    buffer_targets: dict[str, PinnedBufferTarget]


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
        """Pinned host bytes reachable through this module binding.

        This deduplicates repeated binding records that point at the
        same pinned backing inside this module binding. It still is not
        a cross-binding ownership total: two module bindings can share
        the same pinned backing, so callers must not sum this across
        shared-looking bindings unless ownership is known externally.
        """
        total = 0
        seen_params: set[int] = set()
        for param_binding in self.param_bindings:
            key = id(param_binding.pinned)
            if key in seen_params:
                continue
            seen_params.add(key)
            total += param_binding.cache_bytes
        seen_buffers: set[int] = set()
        for buffer_binding in self.buffer_bindings:
            key = id(buffer_binding.pinned)
            if key in seen_buffers:
                continue
            seen_buffers.add(key)
            total += buffer_binding.cache_bytes
        return total

    def contains_param_binding(self, binding: PinnedParamBinding) -> bool:
        return any(binding is candidate for candidate in self.param_bindings)

    def allocate_target(self, device: torch.device) -> PinnedModuleTarget:
        """Allocate active target storage for this binding on ``device``."""
        if device.type != "cuda":
            raise ValueError(
                "PinnedModuleTarget requires a CUDA device; "
                f"got {device}."
            )

        _validate_unique_binding_names(self.param_bindings, kind="param")
        _validate_unique_binding_names(self.buffer_bindings, kind="buffer")

        return PinnedModuleTarget(
            param_targets={
                binding.name: binding.allocate_target(device)
                for binding in self.param_bindings
            },
            buffer_targets={
                binding.name: binding.allocate_target(device)
                for binding in self.buffer_bindings
            },
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
        """Copy pinned storage into ``target`` and set managed slots.

        Slot mutation is deliberately delayed until after all target
        copies and post-copy hooks complete, so a copy failure does not
        leave the module partially active.
        """
        target_params: dict[str, nn.Parameter] = {}
        for param_binding in self.param_bindings:
            param_target = target.param_targets[param_binding.name]
            param_binding.copy_to_target(
                param_target,
                non_blocking=non_blocking,
            )
            target_params[param_binding.name] = param_target.param

        for param_binding in self.param_bindings:
            target_param = target_params[param_binding.name]
            hook = (
                post_copy_hooks.get(id(param_binding))
                if post_copy_hooks is not None
                else None
            )
            if hook is not None:
                hook(target_param)

        target_buffers: dict[str, torch.Tensor] = {}
        for buffer_binding in self.buffer_bindings:
            buffer_target = target.buffer_targets[buffer_binding.name]
            buffer_binding.copy_to_target(
                buffer_target,
                non_blocking=non_blocking,
            )
            target_buffers[buffer_binding.name] = buffer_target.tensor

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
    pinned = PinnedParam(primary_slot.get())
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
    pinned = PinnedBuffer.clone(primary_slot.get())
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
    "PinnedBufferTarget",
    "PinnedModuleBinding",
    "PinnedModuleTarget",
    "PinnedParamBinding",
    "PinnedParamTarget",
    "PostCopyHook",
    "PostCopyHookHandle",
    "pin_module_slot_collection",
]
