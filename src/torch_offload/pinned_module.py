"""Name-based pinned module store and instance primitives.

This module is the migration path for sharing one pinned CPU cache across
multiple concrete model instances. It deliberately does not depend on the
legacy binding/slot layer: names are the only durable relationship
between a store and an instance.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, TypeVar

import torch
from torch import nn

from .pinned_buffer import PinnedBuffer
from .pinned_param import PinnedParam
from .tensor_adapter_factory import storage_key

ParamAliasMode = Literal["storage", "object"]
ParamValidator = Callable[[str, nn.Parameter], None]
_NamedT = TypeVar("_NamedT")


@dataclass(slots=True)
class PinnedModuleStore:
    """Pinned backing bytes for one module layout.

    ``params`` and ``buffers`` are keyed by PyTorch logical names from
    ``named_parameters(remove_duplicate=False)`` and
    ``named_buffers(remove_duplicate=False)``. Aliased names point at the
    same pinned object.
    """

    params: dict[str, PinnedParam]
    buffers: dict[str, PinnedBuffer]
    param_alias_mode: ParamAliasMode = "storage"

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        include_buffers: bool = True,
        param_alias_mode: ParamAliasMode = "storage",
        validate_param: ParamValidator | None = None,
    ) -> PinnedModuleStore:
        """Pin ``module`` into a name-keyed store.

        Store construction is intentionally side-effecting like the
        existing pinning path: after bytes are pinned, the prototype
        module is restored to the store-backed pinned CPU state.
        """
        params = _named_parameters(module)
        if validate_param is not None:
            for name, param in params.items():
                validate_param(name, param)

        store = cls(
            params=_pin_params(params, param_alias_mode),
            buffers=_pin_buffers(_named_buffers(module) if include_buffers else {}),
            param_alias_mode=param_alias_mode,
        )
        _restore_params(module, store.params, _make_cpu_params(store.params))
        _restore_buffers(module, store.buffers)
        return store

    @property
    def cache_bytes(self) -> int:
        return _unique_cache_bytes(self.params) + _unique_cache_bytes(self.buffers)


@dataclass(slots=True)
class PinnedModuleInstance:
    """One concrete module bound to a shared :class:`PinnedModuleStore`."""

    module: nn.Module
    store: PinnedModuleStore
    cpu_params_by_pinned_id: dict[int, nn.Parameter]

    @classmethod
    def from_store(
        cls,
        store: PinnedModuleStore,
        module: nn.Module,
    ) -> PinnedModuleInstance:
        """Validate ``module`` against ``store`` and restore pinned CPU state."""
        _validate_store_names(store)
        _validate_module_matches_store(store, module)
        instance = cls(
            module=module,
            store=store,
            cpu_params_by_pinned_id=_make_cpu_params(store.params),
        )
        instance.restore_pinned()
        return instance

    @property
    def cache_bytes(self) -> int:
        return self.store.cache_bytes

    def restore_pinned(self) -> None:
        _restore_params(
            self.module,
            self.store.params,
            self.cpu_params_by_pinned_id,
        )
        _restore_buffers(self.module, self.store.buffers)


def _pin_params(
    params: Mapping[str, nn.Parameter],
    alias_mode: ParamAliasMode,
) -> dict[str, PinnedParam]:
    pinned_by_name: dict[str, PinnedParam] = {}
    for names in _group_names(
        params,
        lambda name: _param_alias_key(params[name], alias_mode),
    ):
        pinned = PinnedParam(params[names[0]])
        for name in names:
            pinned_by_name[name] = pinned
    return pinned_by_name


def _pin_buffers(buffers: Mapping[str, torch.Tensor]) -> dict[str, PinnedBuffer]:
    pinned_by_name: dict[str, PinnedBuffer] = {}
    for names in _group_names(buffers, lambda name: _buffer_storage_key(buffers[name])):
        pinned = PinnedBuffer.clone(buffers[names[0]])
        for name in names:
            pinned_by_name[name] = pinned
    return pinned_by_name


def _validate_module_matches_store(
    store: PinnedModuleStore, module: nn.Module,
) -> None:
    params = _named_parameters(module)
    buffers = _named_buffers(module)

    _validate_names_present("param", store.params, params)
    _validate_names_present("buffer", store.buffers, buffers)

    for name, pinned in store.params.items():
        param = params[name]
        if param.requires_grad != pinned.requires_grad:
            raise ValueError(
                f"Param {name!r} requires_grad mismatch: store has "
                f"{pinned.requires_grad}, module has {param.requires_grad}."
            )
        layout = PinnedParam.target_layout_for(param)
        if layout != pinned.target_layout:
            raise ValueError(
                f"Param {name!r} layout mismatch: store has "
                f"{pinned.target_layout!r}, module has {layout!r}."
            )

    for name, pinned in store.buffers.items():
        layout = PinnedBuffer.target_layout_for(buffers[name])
        if layout != pinned.target_layout:
            raise ValueError(
                f"Buffer {name!r} layout mismatch: store has "
                f"{pinned.target_layout!r}, module has {layout!r}."
            )

    _validate_alias_topology(
        "param",
        _pinned_alias_groups(store.params),
        _module_param_alias_groups(params, store.param_alias_mode),
    )
    _validate_alias_topology(
        "buffer",
        _pinned_alias_groups(store.buffers),
        _alias_groups(buffers, lambda name: _buffer_storage_key(buffers[name])),
    )


def _validate_store_names(store: PinnedModuleStore) -> None:
    overlap = sorted(set(store.params) & set(store.buffers))
    if overlap:
        raise ValueError(
            "PinnedModuleStore cannot bind names as both params and buffers: "
            f"{_format_names(overlap)}."
        )


def _validate_names_present(
    kind: str,
    store_items: Mapping[str, object],
    module_items: Mapping[str, object],
) -> None:
    missing = sorted(set(store_items) - set(module_items))
    if missing:
        raise ValueError(
            f"Module is missing pinned {kind} names: {_format_names(missing)}."
        )


def _validate_alias_topology(
    kind: str,
    store_groups: Mapping[str, frozenset[str]],
    module_groups: Mapping[str, frozenset[str]],
) -> None:
    mismatched = sorted(
        name
        for name, store_group in store_groups.items()
        if module_groups[name] != store_group
    )
    if not mismatched:
        return

    name = mismatched[0]
    raise ValueError(
        f"Pinned {kind} alias topology mismatch for {_format_names(mismatched)}; "
        f"{name!r} store aliases {sorted(store_groups[name])!r}, "
        f"module aliases {sorted(module_groups[name])!r}."
    )


def _restore_params(
    module: nn.Module,
    params: Mapping[str, PinnedParam],
    cpu_params_by_pinned_id: Mapping[int, nn.Parameter],
) -> None:
    for name, pinned in params.items():
        cpu_param = cpu_params_by_pinned_id[id(pinned)]
        parent, leaf = _resolve_parent_leaf(module, name)
        if pinned.requires_grad:
            pinned.validate_parameter_data_swap_target()
            _get_param(parent, leaf).data = cpu_param.data
        else:
            _set_param(parent, leaf, cpu_param)


def _restore_buffers(
    module: nn.Module,
    buffers: Mapping[str, PinnedBuffer],
) -> None:
    for name, pinned in buffers.items():
        parent, leaf = _resolve_parent_leaf(module, name)
        persistent = leaf not in parent._non_persistent_buffers_set
        parent.register_buffer(leaf, pinned.tensor, persistent=persistent)


def _make_cpu_params(
    params: Mapping[str, PinnedParam],
) -> dict[int, nn.Parameter]:
    return {id(pinned): pinned.make_cpu_param() for pinned in _unique_values(params)}


def _named_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    return _unique_name_dict(
        module.named_parameters(remove_duplicate=False),
        kind="parameter",
    )


def _named_buffers(module: nn.Module) -> dict[str, torch.Tensor]:
    return _unique_name_dict(
        module.named_buffers(remove_duplicate=False),
        kind="buffer",
    )


def _unique_name_dict(
    items: Iterable[tuple[str, _NamedT]],
    *,
    kind: str,
) -> dict[str, _NamedT]:
    values: dict[str, _NamedT] = {}
    for name, value in items:
        if name in values:
            raise ValueError(f"Module yielded duplicate {kind} name {name!r}.")
        values[name] = value
    return values


def _resolve_parent_leaf(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_path, sep, leaf = name.rpartition(".")
    if not sep:
        return root, leaf

    parent = _walk_attr_path(root, parent_path)
    if not isinstance(parent, nn.Module):
        raise TypeError(
            f"Path {parent_path!r} resolved to {type(parent).__name__}, "
            "expected nn.Module."
        )
    return parent, leaf


def _walk_attr_path(root: nn.Module, path: str) -> object:
    current: object = root
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _get_param(parent: nn.Module, leaf: str) -> nn.Parameter:
    param = parent._parameters.get(leaf)
    if param is None:
        raise RuntimeError(f"Parameter {leaf!r} is unexpectedly missing.")
    return param


def _set_param(parent: nn.Module, leaf: str, param: nn.Parameter) -> None:
    if leaf not in parent._parameters:
        raise RuntimeError(f"Parameter {leaf!r} is unexpectedly missing.")
    parent._parameters[leaf] = param


def _module_param_alias_groups(
    params: Mapping[str, nn.Parameter],
    alias_mode: ParamAliasMode,
) -> dict[str, frozenset[str]]:
    if alias_mode == "storage":
        return _alias_groups(params, lambda name: _param_storage_key(params[name]))
    if alias_mode == "object":
        return _alias_groups(params, lambda name: id(params[name]))
    raise ValueError(f"Unsupported param alias mode {alias_mode!r}.")


def _pinned_alias_groups(
    items: Mapping[str, PinnedParam] | Mapping[str, PinnedBuffer],
) -> dict[str, frozenset[str]]:
    return _alias_groups(items, lambda name: id(items[name]))


def _alias_groups(
    items: Mapping[str, object],
    key_for_name: Callable[[str], Hashable],
) -> dict[str, frozenset[str]]:
    groups_by_name: dict[str, frozenset[str]] = {}
    for names in _group_names(items, key_for_name):
        group = frozenset(names)
        for name in names:
            groups_by_name[name] = group
    return groups_by_name


def _group_names(
    items: Mapping[str, object],
    key_for_name: Callable[[str], Hashable],
) -> list[list[str]]:
    groups_by_key: dict[Hashable, list[str]] = {}
    for name in items:
        groups_by_key.setdefault(key_for_name(name), []).append(name)
    return list(groups_by_key.values())


def _param_alias_key(
    param: nn.Parameter, alias_mode: ParamAliasMode,
) -> Hashable:
    if alias_mode == "storage":
        return _param_storage_key(param)
    if alias_mode == "object":
        return id(param)
    raise ValueError(f"Unsupported param alias mode {alias_mode!r}.")


def _param_storage_key(param: nn.Parameter) -> Hashable:
    if param.numel() == 0:
        return ("empty-param", id(param))
    return storage_key(param.data)


def _buffer_storage_key(buffer: torch.Tensor) -> Hashable:
    if buffer.numel() == 0:
        return ("empty-buffer", id(buffer))
    return storage_key(buffer)


def _unique_values(
    items: Mapping[str, PinnedParam],
) -> Iterator[PinnedParam]:
    seen: set[int] = set()
    for value in items.values():
        key = id(value)
        if key in seen:
            continue
        seen.add(key)
        yield value


def _unique_cache_bytes(
    items: Mapping[str, PinnedParam] | Mapping[str, PinnedBuffer],
) -> int:
    total = 0
    seen: set[int] = set()
    for value in items.values():
        key = id(value)
        if key in seen:
            continue
        seen.add(key)
        total += value.cache_bytes
    return total


def _format_names(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in names)


__all__ = [
    "ParamAliasMode",
    "ParamValidator",
    "PinnedModuleInstance",
    "PinnedModuleStore",
]
