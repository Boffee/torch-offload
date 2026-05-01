"""Slot resolution helpers — the single source of truth for walking a
model and producing :class:`SlotOwnership` identities.

A "slot" is a ``(parent_module, leaf_name, kind)`` triple identifying
where a parameter or buffer lives in a module tree (see
:class:`~torch_offload.protocols.SlotOwnership`). The streaming and
pinning components in this package all need to walk a model and resolve
each named parameter/buffer back to its slot. This module owns that walk
so the duplication across ``pinned_weights``, ``streamed_weights``,
and ``model_offloader`` collapses to a single implementation.

The walk uses ``remove_duplicate=False`` throughout: a Parameter or
buffer that's aliased under multiple names yields one row per alias.
Callers that need to dedupe by tensor identity track ``id(...)``
themselves; callers that need every slot covered (e.g. building a
``slot_filter`` that another component will skip) iterate as-is.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn

from .protocols import SlotOwnership

__all__ = [
    "BufferSlot",
    "ParamSlot",
    "iter_buffer_slots",
    "iter_param_slots",
]


@dataclass(slots=True, frozen=True)
class ParamSlot:
    """One row from :func:`iter_param_slots`.

    ``slot`` is the stable identity (survives slot mutation). ``name`` is
    the qualified name from ``named_parameters()`` (may differ across
    aliases of the same Parameter). ``parent`` and ``leaf`` are the live
    references to where the slot lives — useful for callers that mutate
    ``parent._parameters[leaf]`` directly or look up siblings.
    """

    slot: SlotOwnership
    name: str
    param: nn.Parameter
    parent: nn.Module
    leaf: str


@dataclass(slots=True, frozen=True)
class BufferSlot:
    """One row from :func:`iter_buffer_slots`. Mirrors :class:`ParamSlot`
    for the buffer namespace, with ``slot.kind == "buffer"``."""

    slot: SlotOwnership
    name: str
    buffer: torch.Tensor
    parent: nn.Module
    leaf: str


def iter_param_slots(module: nn.Module) -> Iterator[ParamSlot]:
    """Walk every named parameter, yielding alias-aware slot info.

    Uses ``remove_duplicate=False``: a Parameter shared across multiple
    submodule paths yields one :class:`ParamSlot` per name. To dedupe by
    Parameter identity, track ``id(row.param)`` in the consumer.
    """
    modules_map = dict(module.named_modules(remove_duplicate=False))
    for name, p in module.named_parameters(remove_duplicate=False):
        parent, leaf = _resolve_parent_leaf(module, modules_map, name)
        yield ParamSlot(
            slot=SlotOwnership(id(parent), leaf, "param"),
            name=name,
            param=p,
            parent=parent,
            leaf=leaf,
        )


def iter_buffer_slots(module: nn.Module) -> Iterator[BufferSlot]:
    """Walk every named buffer, yielding alias-aware slot info.

    Mirrors :func:`iter_param_slots` for the buffer namespace.
    """
    modules_map = dict(module.named_modules(remove_duplicate=False))
    for name, b in module.named_buffers(remove_duplicate=False):
        parent, leaf = _resolve_parent_leaf(module, modules_map, name)
        yield BufferSlot(
            slot=SlotOwnership(id(parent), leaf, "buffer"),
            name=name,
            buffer=b,
            parent=parent,
            leaf=leaf,
        )


def _resolve_parent_leaf(
    module: nn.Module, modules_map: dict[str, nn.Module], qual_name: str
) -> tuple[nn.Module, str]:
    parts = qual_name.rsplit(".", 1)
    if len(parts) == 2:
        return modules_map[parts[0]], parts[1]
    return module, qual_name
