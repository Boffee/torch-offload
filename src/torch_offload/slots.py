"""Slot resolution helpers.

A "slot" is a ``(parent_module, leaf_name, kind)`` triple identifying
where a parameter or buffer lives in a module tree. Some components need
to walk a model and resolve each named parameter/buffer back to its
owning module slot before mutating module state or validating topology.
This module owns that shared slot walking logic.

The walk uses ``remove_duplicate=False`` throughout: a Parameter or
buffer that's aliased under multiple names yields one row per alias.
Callers that need to dedupe by tensor identity track ``id(...)``
themselves; callers that need every slot covered iterate as-is.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn

from .module_names import canonical_param_name, resolve_parent_leaf, walk_attr_path
from .protocols import SlotKey
from .tensor_adapter_registry import buffer_storage_key, param_storage_key

__all__ = [
    "BufferSlot",
    "ParamSlot",
    "assert_frozen",
    "buffer_storage_key",
    "canonical_param_name",
    "get_param_slot",
    "iter_buffer_slots",
    "iter_param_slots",
    "param_storage_key",
    "set_buffer_slot",
    "set_param_data",
    "set_param_slot",
    "set_tensor_data",
    "split_attr_path",
    "walk_attr_path",
]


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

def assert_frozen(
    slot: "ParamSlot", owner: str, *, extra: str | None = None,
) -> None:
    """Raise if ``slot`` is trainable.

    Slot-replacing strategies must keep trainable params out of this
    path. Replacing the Parameter object orphans optimizer state keyed
    by the user's pre-wrap Parameter and breaks grad identity, so
    trainable slots must be partitioned out by the caller. Fail loud
    rather than silently freeze.

    ``owner`` is surfaced in the error message (e.g.
    ``"PinnedComponent"``, ``"StreamedComponent"``). ``extra`` is appended
    verbatim for owner-specific recovery guidance.
    """
    if not slot.get().requires_grad:
        return
    msg = (
        f"{owner} cannot manage trainable slot {slot.name!r}: slot "
        "replacement installs a fresh frozen Parameter wrapper, "
        "breaking optimizer/grad identity. Use ModelOffloader, which "
        "preserves trainable Parameter identity."
    )
    if extra:
        msg = f"{msg} {extra}"
    raise ValueError(msg)


def get_param_slot(parent: nn.Module, leaf: str) -> nn.Parameter:
    """Return the live Parameter registered at ``parent._parameters[leaf]``.

    Slot-mutating strategies call this before identity-preserving
    ``.data`` swaps. A missing/empty slot means strategy bookkeeping has
    drifted from the module tree, so fail with a clear internal error.
    """
    param = parent._parameters[leaf]
    if param is None:
        raise RuntimeError(f"Parameter slot {leaf!r} is unexpectedly empty")
    return param


def set_param_slot(parent: nn.Module, leaf: str, param: nn.Parameter) -> None:
    """Replace a module parameter slot with ``param``.

    This intentionally uses ``module._parameters`` rather than
    ``setattr`` so tensor subclasses keep their wrapper state and tied
    slots can be assigned the same ``Parameter`` object.
    """
    parent._parameters[leaf] = param


def set_buffer_slot(
    parent: nn.Module,
    leaf: str,
    buffer: torch.Tensor,
    *,
    persistent: bool,
) -> None:
    """Replace a registered buffer while preserving its persistence flag."""
    parent.register_buffer(leaf, buffer, persistent=persistent)


def set_param_data(param: nn.Parameter, data: torch.Tensor) -> None:
    """Repoint a Parameter's storage while preserving object identity."""
    param.data = data


def set_tensor_data(tensor: torch.Tensor, data: torch.Tensor) -> None:
    """Repoint a Tensor-like buffer's storage while preserving object identity."""
    tensor.data = data


@dataclass(slots=True, frozen=True)
class ParamSlot:
    """One named parameter slot from :func:`iter_param_slots`.

    ``name`` is the qualified name from ``named_parameters()``. ``parent``
    and ``leaf`` identify the live module registry entry, so the slot
    remains useful after the Parameter object installed there changes.
    """

    name: str
    parent: nn.Module
    leaf: str

    @property
    def key(self) -> SlotKey:
        return SlotKey(id(self.parent), self.leaf, "param")

    def get(self) -> nn.Parameter:
        return get_param_slot(self.parent, self.leaf)

    def set(self, param: nn.Parameter) -> None:
        set_param_slot(self.parent, self.leaf, param)


@dataclass(slots=True, frozen=True)
class BufferSlot:
    """One named buffer slot from :func:`iter_buffer_slots`."""

    name: str
    parent: nn.Module
    leaf: str
    persistent: bool

    @property
    def key(self) -> SlotKey:
        return SlotKey(id(self.parent), self.leaf, "buffer")

    def get(self) -> torch.Tensor:
        buffer = self.parent._buffers[self.leaf]
        if buffer is None:
            raise RuntimeError(f"Buffer slot {self.leaf!r} is unexpectedly empty")
        return buffer

    def set(self, buffer: torch.Tensor) -> None:
        set_buffer_slot(
            self.parent,
            self.leaf,
            buffer,
            persistent=self.persistent,
        )


def iter_param_slots(module: nn.Module) -> Iterator[ParamSlot]:
    """Walk every named parameter, yielding alias-aware slot info.

    Uses ``remove_duplicate=False``: a Parameter shared across multiple
    submodule paths yields one :class:`ParamSlot` per name. To dedupe by
    Parameter identity, track ``id(row.get())`` in the consumer.
    """
    for name, _p in module.named_parameters(remove_duplicate=False):
        parent, leaf = resolve_parent_leaf(module, name)
        yield ParamSlot(
            name=name,
            parent=parent,
            leaf=leaf,
        )


def iter_buffer_slots(module: nn.Module) -> Iterator[BufferSlot]:
    """Walk every named buffer, yielding alias-aware slot info.

    Mirrors :func:`iter_param_slots` for the buffer namespace.
    """
    for name, _b in module.named_buffers(remove_duplicate=False):
        parent, leaf = resolve_parent_leaf(module, name)
        yield BufferSlot(
            name=name,
            parent=parent,
            leaf=leaf,
            persistent=leaf not in parent._non_persistent_buffers_set,
        )
