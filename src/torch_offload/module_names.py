"""Name-based helpers for walking and mutating ``nn.Module`` trees."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator

import torch
from torch import nn


def walk_attr_path(root: nn.Module, dotted_path: str) -> object:
    """Walk a dotted attribute path from ``root``."""
    obj: object = root
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_parent_leaf(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    """Resolve a PyTorch-qualified tensor name to its parent module and leaf."""
    parent_path, separator, leaf = name.rpartition(".")
    if not separator:
        return root, leaf

    parent = walk_attr_path(root, parent_path)
    if not isinstance(parent, nn.Module):
        raise TypeError(
            f"Path {parent_path!r} resolved to {type(parent).__name__}, "
            "expected nn.Module."
        )
    return parent, leaf


def parameter_names(module: nn.Module) -> set[str]:
    return {
        name
        for name, _param in module.named_parameters(remove_duplicate=False)
    }


def buffer_names(module: nn.Module) -> set[str]:
    return {
        name
        for name, _buffer in module.named_buffers(remove_duplicate=False)
    }


def named_parameter_entries(
    module: nn.Module,
) -> Iterator[tuple[str, nn.Module, str, nn.Parameter]]:
    for name, param in module.named_parameters(remove_duplicate=False):
        parent, leaf = resolve_parent_leaf(module, name)
        yield name, parent, leaf, param


def named_buffer_entries(
    module: nn.Module,
) -> Iterator[tuple[str, nn.Module, str, torch.Tensor, bool]]:
    for name, buffer in module.named_buffers(remove_duplicate=False):
        parent, leaf = resolve_parent_leaf(module, name)
        persistent = leaf not in parent._non_persistent_buffers_set
        yield name, parent, leaf, buffer, persistent


def set_named_parameter(
    parent: nn.Module,
    leaf: str,
    param: nn.Parameter,
) -> None:
    parent._parameters[leaf] = param


def set_named_buffer(
    parent: nn.Module,
    leaf: str,
    buffer: torch.Tensor,
    *,
    persistent: bool,
) -> None:
    parent.register_buffer(leaf, buffer, persistent=persistent)


def group_names(
    names: Iterable[str],
    key_for_name: Callable[[str], Hashable],
) -> list[tuple[str, ...]]:
    groups_by_key: dict[Hashable, list[str]] = {}
    for name in names:
        groups_by_key.setdefault(key_for_name(name), []).append(name)
    return [tuple(group) for group in groups_by_key.values()]
