"""Shared pinned binding records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .pinned_param import PinnedParam
from .slots import BufferSlot, ParamSlot, unique_slots
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


def bind_param_slots(
    pinned: PinnedParam, slots: Sequence[ParamSlot],
) -> PinnedParamBinding:
    """Bind live model slots to an existing pinned parameter backing."""
    slot_list = list(slots)
    if not slot_list:
        raise ValueError("bind_param_slots requires at least one ParamSlot")
    return PinnedParamBinding(
        pinned=pinned,
        slots=slot_list,
        cpu_param=pinned.make_cpu_param(),
    )


def pin_param_slots(slots: Sequence[ParamSlot]) -> PinnedParamBinding:
    """Pin the first slot's parameter and bind all aliases to that backing."""
    slot_list = list(slots)
    if not slot_list:
        raise ValueError("pin_param_slots requires at least one ParamSlot")
    primary_slot = slot_list[0]
    pinned = PinnedParam(primary_slot.name, primary_slot.get())
    return bind_param_slots(pinned, slot_list)


def pin_buffer_slots(slots: Sequence[BufferSlot]) -> PinnedBufferBinding:
    """Clone and pin the first slot's buffer and bind all aliases to it."""
    slot_list = list(slots)
    if not slot_list:
        raise ValueError("pin_buffer_slots requires at least one BufferSlot")
    pinned = clone_to_pinned_cpu(
        slot_list[0].get(),
        memory_format=torch.contiguous_format,
    )
    return PinnedBufferBinding(pinned=pinned, slots=slot_list)


__all__ = [
    "PinnedBufferBinding",
    "PinnedParamBinding",
    "bind_param_slots",
    "pin_buffer_slots",
    "pin_param_slots",
]
