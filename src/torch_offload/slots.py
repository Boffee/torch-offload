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
    "assert_frozen",
    "canonical_param_name",
    "iter_buffer_slots",
    "iter_param_slots",
    "split_attr_path",
    "walk_attr_path",
]


def walk_attr_path(root: nn.Module, dotted_path: str) -> object:
    """Walk a dotted attribute path from ``root``, returning the leaf.

    Equivalent to a chained ``getattr``. Used to resolve dotted paths
    like ``"transformer.blocks"`` or ``"a.b.weight"`` to their target.
    Raises ``AttributeError`` if any segment is missing.
    """
    obj: object = root
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def split_attr_path(
    root: nn.Module, dotted_path: str,
) -> tuple[nn.Module, str]:
    """Resolve all but the last segment, returning ``(parent, leaf)``.

    Useful for in-place mutation: ``parent, leaf = split_attr_path(...);
    setattr(parent, leaf, ...)``. Raises ``ValueError`` for a top-level
    path (no parent to mutate).
    """
    parts = dotted_path.rsplit(".", 1)
    if len(parts) == 1:
        raise ValueError(f"Cannot split top-level path {dotted_path!r}")
    parent_path, leaf = parts
    parent = walk_attr_path(root, parent_path)
    if not isinstance(parent, nn.Module):
        raise TypeError(
            f"Path '{parent_path}' resolved to {type(parent).__name__}, "
            "expected nn.Module"
        )
    return parent, leaf


def canonical_param_name(name: str) -> str:
    """Normalize a parameter name to its canonical (non-PEFT) form.

    PEFT inserts ``.base_layer.`` into wrapped module paths
    (e.g. ``to_q.base_layer.weight`` instead of ``to_q.weight``).
    LoRA state dicts always use the original names, so target maps
    built from named-parameter walks must store canonical keys for
    matching to work.
    """
    return name.replace(".base_layer.", ".")


def assert_frozen(
    slot: "ParamSlot", owner: str, *, extra: str | None = None,
) -> None:
    """Raise if ``slot`` is trainable.

    Both :class:`PinnedWeights` and :class:`StreamedWeights` replace
    the Parameter object at every slot they manage. That orphans
    optimizer state keyed by the user's pre-wrap Parameter and breaks
    grad identity, so trainable slots must be partitioned out by the
    caller. The composer (:class:`ModelOffloader`) does this
    automatically via ``skip_slots``; direct users are on the hook.
    Fail loud rather than silently freeze.

    ``owner`` is surfaced in the error message (e.g.
    ``"PinnedWeights"``, ``"StreamedWeights"``). ``extra`` is appended
    verbatim for owner-specific recovery guidance.
    """
    if not slot.param.requires_grad:
        return
    msg = (
        f"{owner} cannot manage trainable slot {slot.name!r}: slot "
        "replacement installs a fresh frozen Parameter wrapper, "
        "breaking optimizer/grad identity. Use ModelOffloader (which "
        "partitions trainables into TrainableWeights automatically), "
        "or pass the slot in skip_slots and route it to a separate "
        "trainable mover."
    )
    if extra:
        msg = f"{msg} {extra}"
    raise ValueError(msg)


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
