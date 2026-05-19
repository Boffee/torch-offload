"""Module slot collection and alias grouping helpers.

``slots.py`` owns the low-level PyTorch registry access. This module
owns the next layer up: walking a module, applying skip filters, and
grouping parameter/buffer aliases before a caller decides how to pin,
stream, merge, or validate those slots.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from .pinned_param import storage_key
from .protocols import SlotKey
from .slots import BufferSlot, ParamSlot, iter_buffer_slots, iter_param_slots

ParamGroupBy = Literal["storage", "object"]


@dataclass(slots=True)
class ModuleSlotCollection:
    """Alias-aware slots collected from one module scope."""

    param_slot_groups: list[list[ParamSlot]]
    buffer_slot_groups: list[list[BufferSlot]]
    slot_filter: frozenset[SlotKey]


def param_storage_key(param: nn.Parameter) -> tuple[Any, ...]:
    """Return a storage-identity grouping key for a parameter."""
    if param.numel() == 0:
        # Zero-sized tensors all share data_ptr()==0; key by object
        # identity so aliases of the same Parameter still dedupe.
        return ("__empty__", id(param))
    return storage_key(param.data)


def buffer_storage_key(buffer: torch.Tensor) -> tuple[Any, ...]:
    """Return a storage-identity grouping key for a registered buffer."""
    if buffer.numel() == 0:
        return ("__empty_buf__", id(buffer))
    return storage_key(buffer)


def collect_module_slots(
    module: nn.Module,
    *,
    skip_slots: set[SlotKey] | frozenset[SlotKey] | None = None,
    include_buffers: bool = True,
    param_group_by: ParamGroupBy = "storage",
    validate_param: Callable[[ParamSlot], None] | None = None,
) -> ModuleSlotCollection:
    """Collect grouped parameter and buffer slots from ``module``.

    Parameters are grouped either by storage identity (for whole-module
    pinning, where tied frozen weights must share one pinned backing) or
    by ``Parameter`` object identity (for direct streamed blocks, where
    composer-level validation owns distinct-Parameter storage ties).
    Buffers are always grouped by storage identity.
    """
    skip = skip_slots or frozenset()
    slot_filter: set[SlotKey] = set()

    param_groups: dict[tuple[Any, ...], list[ParamSlot]] = {}
    for slot in iter_param_slots(module):
        if slot.key in skip:
            continue
        if validate_param is not None:
            validate_param(slot)
        slot_filter.add(slot.key)
        param = slot.get()
        param_key = _param_group_key(param, param_group_by)
        param_groups.setdefault(param_key, []).append(slot)

    buffer_groups: dict[tuple[Any, ...], list[BufferSlot]] = {}
    if include_buffers:
        for slot in iter_buffer_slots(module):
            if slot.key in skip:
                continue
            slot_filter.add(slot.key)
            buffer_groups.setdefault(
                buffer_storage_key(slot.get()), []
            ).append(slot)

    return ModuleSlotCollection(
        param_slot_groups=list(param_groups.values()),
        buffer_slot_groups=list(buffer_groups.values()),
        slot_filter=frozenset(slot_filter),
    )


def _param_group_key(
    param: nn.Parameter, group_by: ParamGroupBy,
) -> tuple[Any, ...]:
    if group_by == "storage":
        return param_storage_key(param)
    return ("__param_object__", id(param))


__all__ = [
    "ModuleSlotCollection",
    "ParamGroupBy",
    "buffer_storage_key",
    "collect_module_slots",
    "param_storage_key",
]
