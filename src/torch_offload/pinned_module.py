"""Name-based pinned module store and instance primitives.

This module supports sharing one pinned CPU cache across multiple
concrete model instances. Names are the durable relationship between a
store and an instance.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from dataclasses import dataclass, field
from typing import TypeVar

import torch
from torch import nn

from .module_names import group_names, resolve_parent_leaf
from .pinned_buffer import PinnedBuffer
from .pinned_param import PinnedParam
from .tensor_adapter_registry import buffer_tensor_id, param_tensor_id

PostCopyHook = Callable[[nn.Parameter], None]
_NamedT = TypeVar("_NamedT")


@dataclass(slots=True)
class PinnedParamTarget:
    """Active adapter storage for one pinned parameter backing."""

    _state: object
    param: nn.Parameter


@dataclass(slots=True)
class PinnedBufferTarget:
    """Active tensor storage for one pinned buffer backing."""

    tensor: torch.Tensor


@dataclass(slots=True)
class PinnedModuleTarget:
    """Name-keyed active storage for a :class:`PinnedModuleStore`.

    Targets may contain the whole store or a validated subset of it.
    Names mapped to the same pinned object also point at the same
    target object.
    """

    param_targets: dict[str, PinnedParamTarget]
    buffer_targets: dict[str, PinnedBufferTarget]


class PostCopyHookHandle:
    """Removal handle returned by post-copy hook registration."""

    __slots__ = ("_hooks", "_key")

    def __init__(
        self,
        hooks: MutableMapping[int, PostCopyHook],
        key: int,
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
class PinnedModuleStore:
    """Pinned backing bytes for one module layout.

    ``params`` and ``buffers`` are keyed by PyTorch logical names from
    ``named_parameters(remove_duplicate=False)`` and
    ``named_buffers(remove_duplicate=False)``. Selected names sharing
    storage point at the same pinned object.
    """

    params: dict[str, PinnedParam]
    buffers: dict[str, PinnedBuffer]

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        include_param_names: Iterable[str] | None = None,
        include_buffer_names: Iterable[str] | None = None,
    ) -> PinnedModuleStore:
        """Pin ``module`` into a name-keyed store.

        Store construction is intentionally side-effecting like the
        existing pinning path: after bytes are pinned, selected module
        state is restored to the store-backed pinned CPU state.
        """
        all_params = _named_parameters(module)
        params = _select_known_names(
            all_params,
            include_param_names,
        )

        all_buffers = _named_buffers(module)
        buffers = _select_known_names(
            all_buffers,
            include_buffer_names,
        )

        store = cls(
            params=_pin_params(params),
            buffers=_pin_buffers(buffers),
        )
        _validate_trainable_param_data_swaps(store.params)
        _install_pinned_params(module, store.params)
        _install_pinned_buffers(module, store.buffers)
        return store

    @property
    def cache_bytes(self) -> int:
        return _unique_cache_bytes(self.params) + _unique_cache_bytes(self.buffers)

    @property
    def has_trainables(self) -> bool:
        return bool(self.trainable_param_names)

    @property
    def trainable_param_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, pinned in self.params.items()
            if pinned.requires_grad
        )

    def bind(self, module: nn.Module) -> PinnedModuleInstance:
        """Validate ``module`` and bind this store's backing bytes to it.

        The sole instance factory: layout-checks ``module`` against this
        store, constructs a :class:`PinnedModuleInstance` that owns
        ``module`` and shares this store's pinned host bytes, then installs
        those pinned bytes onto ``module`` via
        :meth:`PinnedModuleInstance.install_pinned`.
        """
        _validate_module_matches(self.params, self.buffers, module)
        instance = PinnedModuleInstance(
            module=module,
            params=self.params,
            buffers=self.buffers,
        )
        instance.install_pinned()
        return instance


@dataclass(slots=True)
class PinnedModuleInstance:
    """One concrete module bound to pinned parameter and buffer backings.

    Owns the :class:`nn.Module` whose managed params and buffers are
    backed by this instance's pinned host bytes, plus the copy machinery
    to stream them to (and trainable bytes back from) a GPU
    :class:`PinnedModuleTarget`. :meth:`load_to_target` copies into a
    target and installs that active storage onto :attr:`module`;
    :meth:`install_pinned` installs the pinned host bytes onto
    :attr:`module`. The pure-copy method :meth:`copy_trainables_from_target`
    touches no module at all.
    """

    module: nn.Module
    params: Mapping[str, PinnedParam]
    buffers: Mapping[str, PinnedBuffer]
    _post_copy_hooks: dict[int, PostCopyHook] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def has_trainables(self) -> bool:
        return bool(self.trainable_param_names)

    @property
    def trainable_param_names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, pinned in self.params.items()
            if pinned.requires_grad
        )

    def install_pinned(self) -> None:
        """Install the pinned host bytes onto :attr:`module`'s attributes.

        Pure pinned-CPU repoint: materializes the pinned CPU wrappers on
        demand (deduped by ``id(pinned)`` so tied names share one wrapper)
        and installs them onto :attr:`module`, leaving its managed state on
        the pinned host bytes. Mutates :attr:`module` in place.
        """
        _install_pinned_params(self.module, self.params)
        _install_pinned_buffers(self.module, self.buffers)

    def allocate_target(
        self,
        device: torch.device,
        *,
        param_names: Iterable[str] | None = None,
        buffer_names: Iterable[str] | None = None,
    ) -> PinnedModuleTarget:
        """Allocate active storage for selected bound entries on ``device``."""
        _validate_cuda_device(device)
        params = _select_known_names(self.params, param_names)
        buffers = _select_known_names(self.buffers, buffer_names)
        return PinnedModuleTarget(
            param_targets=_allocate_param_targets(params, device),
            buffer_targets=_allocate_buffer_targets(buffers, device),
        )

    def register_post_copy_hook(
        self, name: str, hook: PostCopyHook,
    ) -> PostCopyHookHandle:
        """Register a hook after this instance copies ``name`` to a target."""
        key = self.post_copy_hook_key(name)
        if key in self._post_copy_hooks:
            raise RuntimeError(
                "post-copy hook already registered for "
                f"param name {name!r}. Duplicate or shared LoRA "
                "targets for the same parameter backing are unsupported."
            )
        self._post_copy_hooks[key] = hook
        return PostCopyHookHandle(self._post_copy_hooks, key)

    def post_copy_hook_key(self, name: str) -> int:
        """Stable hook/dedup key for a managed parameter name."""
        if name not in self.params:
            raise ValueError(
                f"param name {name!r} is not owned by this PinnedModuleInstance"
            )
        return id(self.params[name])

    def load_to_target(
        self,
        target: PinnedModuleTarget,
        *,
        run_post_copy_hooks: bool = False,
        non_blocking: bool = False,
    ) -> None:
        """Copy selected pinned bytes into ``target`` and install them.

        Copies the selected pinned params and buffers into ``target``,
        runs any registered post-copy hooks, then repoints :attr:`module`'s
        managed attributes at the filled target storage. Copying and hooks
        complete before any module mutation, so a copy failure does not
        leave :attr:`module` partially active.
        """
        _validate_target_names_known(self.params, self.buffers, target)
        params = _items_for_names(self.params, target.param_targets)
        buffers = _items_for_names(self.buffers, target.buffer_targets)

        _copy_params_to_target(
            params,
            target.param_targets,
            non_blocking=non_blocking,
        )
        if run_post_copy_hooks:
            _run_post_copy_hooks(
                params,
                target.param_targets,
                self._post_copy_hooks,
            )
        _copy_buffers_to_target(
            buffers,
            target.buffer_targets,
            non_blocking=non_blocking,
        )

        _set_params(
            self.module,
            {
                name: param_target.param
                for name, param_target in target.param_targets.items()
            },
        )
        _set_buffers(
            self.module,
            {
                name: buffer_target.tensor
                for name, buffer_target in target.buffer_targets.items()
            },
        )

    def copy_trainables_from_target(
        self,
        target: PinnedModuleTarget,
        *,
        non_blocking: bool = False,
    ) -> None:
        """Copy trainable target params back into pinned host storage.

        This is the explicit pinned-cache mutation path for optimizer-step
        sync. Frozen params and buffers are intentionally not copied back.
        """
        _validate_target_names_known(self.params, self.buffers, target)
        _validate_target_has_trainable_params(self.params, target)
        _copy_trainable_params_from_target(
            self.params,
            target.param_targets,
            non_blocking=non_blocking,
        )

    def move_trainable_grads_to(self, device: torch.device) -> None:
        """Move each trainable param's ``.grad`` (if any) to ``device``.

        During backward, PyTorch's native ``AccumulateGrad`` writes grads on
        the param's data device. As ``.data`` is moved between pinned CPU and
        a GPU target, ``.grad`` keeps living wherever ``AccumulateGrad`` placed
        it; this realigns grad with data so the optimizer reads both on the
        same device. Tied params are deduplicated, and ``None`` grads (no
        backward yet, or ``zero_grad(set_to_none=True)``) are skipped.
        """
        for param in self._iter_trainable_params():
            grad = param.grad
            if grad is None or grad.device == device:
                continue
            moved = grad.to(device)
            if param.data.device == device:
                param.grad = moved
            else:
                # PyTorch's grad setter rejects cross-device grad/data pairs.
                # A trainable can transiently have offloaded (pinned-CPU) data
                # and a GPU grad, so move the grad storage in place instead.
                grad.data = moved.data

    def _iter_trainable_params(self) -> Iterator[nn.Parameter]:
        params = dict(self.module.named_parameters(remove_duplicate=False))
        seen: set[int] = set()
        for name, pinned in self.params.items():
            if not pinned.requires_grad:
                continue
            param = params[name]
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            yield param


def _pin_params(params: Mapping[str, nn.Parameter]) -> dict[str, PinnedParam]:
    pinned_by_name: dict[str, PinnedParam] = {}
    for names in group_names(
        params.keys(),
        lambda name: param_tensor_id(params[name]),
    ):
        _validate_param_storage_group_requires_grad(names, params)
        pinned = PinnedParam(params[names[0]])
        _validate_param_storage_group_tieable(names, pinned)
        for name in names:
            pinned_by_name[name] = pinned
    return pinned_by_name


def _validate_param_storage_group_tieable(
    names: Sequence[str], pinned: PinnedParam
) -> None:
    """Reject tied weights whose adapter migrates wrapper state on forward.

    A migrate-state adapter (bitsandbytes int8) shares one reconstructed
    wrapper across all tied leaves, but the first module's forward migrates
    its quant state onto that module and nulls it on the wrapper — so a
    second tied module computes against missing state (garbage or a crash).
    The per-load rearm cannot fix this: it fires once per load, not once per
    consuming module. ``_needs_rearm`` flags exactly these adapters.
    """
    if len(names) > 1 and pinned._needs_rearm:
        raise NotImplementedError(
            f"Tied weights {sorted(names)!r} use an adapter whose quant state "
            f"migrates onto the owning module on first forward "
            f"({type(pinned.adapter).__name__}); one shared wrapper cannot "
            "serve multiple tied modules. Untie these weights, or keep them "
            "resident instead of offloading."
        )


def _pin_buffers(buffers: Mapping[str, torch.Tensor]) -> dict[str, PinnedBuffer]:
    pinned_by_name: dict[str, PinnedBuffer] = {}
    for names in group_names(
        buffers.keys(),
        lambda name: buffer_tensor_id(buffers[name]),
    ):
        pinned = PinnedBuffer.clone(buffers[names[0]])
        for name in names:
            pinned_by_name[name] = pinned
    return pinned_by_name


def _select_known_names(
    items: Mapping[str, _NamedT],
    names: Iterable[str] | None,
) -> dict[str, _NamedT]:
    if names is None:
        return dict(items)

    included = set(names)
    missing = sorted(included - set(items))
    if missing:
        raise ValueError(f"Cannot select unknown names: {_format_names(missing)}.")
    return {name: value for name, value in items.items() if name in included}


def _items_for_names(
    items: Mapping[str, _NamedT],
    names: Iterable[str],
) -> dict[str, _NamedT]:
    included = set(names)
    return {name: value for name, value in items.items() if name in included}


def _validate_module_matches(
    pinned_params: Mapping[str, PinnedParam],
    pinned_buffers: Mapping[str, PinnedBuffer],
    module: nn.Module,
) -> None:
    """Validate that ``module`` is a structurally compatible bind target.

    Compares bind layouts, not full target layouts: binding replaces every
    managed tensor with the pinned backing storage, so placeholder fields
    the bind overwrites (dtype, for plain tensors) are not required to
    match — a config-built meta skeleton binds against bytes pinned from
    natively loaded weights.
    """
    params = _named_parameters(module)
    buffers = _named_buffers(module)

    _validate_names_present(pinned_params, pinned_buffers, params, buffers)

    for name, pinned in pinned_params.items():
        param = params[name]
        if param.requires_grad != pinned.requires_grad:
            raise ValueError(
                f"Param {name!r} requires_grad mismatch: store has "
                f"{pinned.requires_grad}, module has {param.requires_grad}."
            )
        layout = PinnedParam.bind_layout_for(param)
        if layout != pinned.bind_layout:
            raise ValueError(
                f"Param {name!r} layout mismatch: store has "
                f"{pinned.bind_layout!r}, module has {layout!r}."
            )

    for name, pinned in pinned_buffers.items():
        layout = PinnedBuffer.bind_layout_for(buffers[name])
        if layout != PinnedBuffer.bind_layout_for(pinned.tensor):
            raise ValueError(
                f"Buffer {name!r} layout mismatch: store has "
                f"{PinnedBuffer.bind_layout_for(pinned.tensor)!r}, module has {layout!r}."
            )


def _validate_target_has_trainable_params(
    params: Mapping[str, PinnedParam],
    target: PinnedModuleTarget,
) -> None:
    trainable_params = _trainable_params(params)
    expected_names = set(trainable_params)
    actual_names = set(target.param_targets)
    missing = sorted(expected_names - actual_names)
    if missing:
        raise ValueError(
            "PinnedModuleTarget trainable param target names mismatch: "
            f"missing {_format_names(missing)}."
        )


def _validate_target_names_known(
    params: Mapping[str, PinnedParam],
    buffers: Mapping[str, PinnedBuffer],
    target: PinnedModuleTarget,
) -> None:
    extra_params = sorted(set(target.param_targets) - set(params))
    extra_buffers = sorted(set(target.buffer_targets) - set(buffers))
    if not extra_params and not extra_buffers:
        return

    details = []
    if extra_params:
        details.append(f"params {_format_names(extra_params)}")
    if extra_buffers:
        details.append(f"buffers {_format_names(extra_buffers)}")
    raise ValueError(
        "PinnedModuleTarget contains entries outside the store: "
        f"{'; '.join(details)}."
    )


def _validate_param_storage_group_requires_grad(
    names: Iterable[str],
    params: Mapping[str, nn.Parameter],
) -> None:
    names = list(names)
    requires_grad = {params[name].requires_grad for name in names}
    if len(requires_grad) <= 1:
        return
    raise ValueError(
        "PinnedModuleStore cannot group params with mixed requires_grad: "
        f"{_format_names(names)}."
    )


def _validate_trainable_param_data_swaps(
    params: Mapping[str, PinnedParam],
) -> None:
    seen: set[int] = set()
    for name, pinned in params.items():
        if not pinned.requires_grad:
            continue
        key = id(pinned)
        if key in seen:
            continue
        seen.add(key)
        try:
            pinned.validate_parameter_data_swap_target()
        except NotImplementedError as exc:
            raise NotImplementedError(
                f"Trainable param {name!r} cannot use Parameter.data swap: {exc}"
            ) from exc


def _validate_names_present(
    pinned_params: Mapping[str, PinnedParam],
    pinned_buffers: Mapping[str, PinnedBuffer],
    params: Mapping[str, nn.Parameter],
    buffers: Mapping[str, torch.Tensor],
) -> None:
    missing_params = sorted(set(pinned_params) - set(params))
    missing_buffers = sorted(set(pinned_buffers) - set(buffers))
    if not missing_params and not missing_buffers:
        return

    details = []
    if missing_params:
        details.append(f"params {_format_names(missing_params)}")
    if missing_buffers:
        details.append(f"buffers {_format_names(missing_buffers)}")
    raise ValueError(f"Module is missing pinned names: {'; '.join(details)}.")


def _allocate_param_targets(
    params: Mapping[str, PinnedParam],
    device: torch.device,
) -> dict[str, PinnedParamTarget]:
    targets_by_pinned_id: dict[int, PinnedParamTarget] = {}
    targets_by_name: dict[str, PinnedParamTarget] = {}
    for name, pinned in params.items():
        key = id(pinned)
        target = targets_by_pinned_id.get(key)
        if target is None:
            state = pinned.allocate_gpu_storage(device)
            target = PinnedParamTarget(
                _state=state,
                param=pinned.make_gpu_param(state),
            )
            targets_by_pinned_id[key] = target
        targets_by_name[name] = target
    return targets_by_name


def _validate_cuda_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError(
            "PinnedModuleTarget requires a CUDA device; "
            f"got {device}."
        )


def _trainable_params(
    params: Mapping[str, PinnedParam],
) -> dict[str, PinnedParam]:
    return {
        name: pinned
        for name, pinned in params.items()
        if pinned.requires_grad
    }


def _allocate_buffer_targets(
    buffers: Mapping[str, PinnedBuffer],
    device: torch.device,
) -> dict[str, PinnedBufferTarget]:
    targets_by_pinned_id: dict[int, PinnedBufferTarget] = {}
    targets_by_name: dict[str, PinnedBufferTarget] = {}
    for name, pinned in buffers.items():
        key = id(pinned)
        target = targets_by_pinned_id.get(key)
        if target is None:
            target = PinnedBufferTarget(
                tensor=torch.empty_like(pinned.tensor, device=device),
            )
            targets_by_pinned_id[key] = target
        targets_by_name[name] = target
    return targets_by_name


def _copy_params_to_target(
    params: Mapping[str, PinnedParam],
    targets: Mapping[str, PinnedParamTarget],
    *,
    non_blocking: bool,
) -> None:
    copied: set[int] = set()
    for name, pinned in params.items():
        key = id(pinned)
        if key in copied:
            continue
        pinned.copy_to_gpu(targets[name]._state, non_blocking=non_blocking)
        # Re-arm the reused wrapper at the freshly-loaded buffers (no-op
        # unless the adapter migrates state off the wrapper, e.g. bnb int8).
        pinned.rearm_after_load(targets[name].param, targets[name]._state)
        copied.add(key)


def _copy_buffers_to_target(
    buffers: Mapping[str, PinnedBuffer],
    targets: Mapping[str, PinnedBufferTarget],
    *,
    non_blocking: bool,
) -> None:
    copied: set[int] = set()
    for name, pinned in buffers.items():
        key = id(pinned)
        if key in copied:
            continue
        targets[name].tensor.copy_(pinned.tensor, non_blocking=non_blocking)
        copied.add(key)


def _copy_trainable_params_from_target(
    params: Mapping[str, PinnedParam],
    targets: Mapping[str, PinnedParamTarget],
    *,
    non_blocking: bool,
) -> None:
    copied: set[int] = set()
    for name, pinned in params.items():
        if not pinned.requires_grad:
            continue
        key = id(pinned)
        if key in copied:
            continue
        pinned.copy_to_cpu(targets[name]._state, non_blocking=non_blocking)
        copied.add(key)


def _run_post_copy_hooks(
    params: Mapping[str, PinnedParam],
    targets: Mapping[str, PinnedParamTarget],
    hooks: Mapping[int, PostCopyHook],
) -> None:
    seen: set[int] = set()
    for name, pinned in params.items():
        key = id(pinned)
        if key in seen:
            continue
        seen.add(key)
        hook = hooks.get(key)
        if hook is not None:
            hook(targets[name].param)


def _install_pinned_params(
    module: nn.Module,
    params: Mapping[str, PinnedParam],
) -> None:
    # Build the materialized CPU params on demand, deduped by ``id(pinned)``
    # so tied names share one wrapper (preserving tied-weight behavior).
    # ``make_cpu_param`` is cheap/zero-copy for every adapter (a plain
    # wrapper / metadata reconstruction aliasing the pinned tensors), so
    # per-install construction is fine.
    materialized: dict[str, nn.Parameter] = {}
    by_pinned: dict[int, nn.Parameter] = {}
    for name, pinned in params.items():
        cpu_param = by_pinned.get(id(pinned))
        if cpu_param is None:
            cpu_param = pinned.make_cpu_param()
            by_pinned[id(pinned)] = cpu_param
        materialized[name] = cpu_param
    _set_params(module, materialized)


def _set_params(
    module: nn.Module,
    materialized_params: Mapping[str, nn.Parameter],
) -> None:
    # Both materialized sources carry the correct ``requires_grad`` (GPU
    # param via ``make_gpu_param``, CPU wrapper via ``make_cpu_param``), so
    # the swap-vs-replace decision reads off the materialized param itself.
    for name, materialized in materialized_params.items():
        parent, leaf = resolve_parent_leaf(module, name)
        if materialized.requires_grad:
            # Trainable: keep the user's wrapper, swap only ``.data``.
            _get_param(parent, leaf).data = materialized.data
        else:
            # Frozen: replace the registry entry outright.
            _set_param(parent, leaf, materialized)


def _install_pinned_buffers(
    module: nn.Module,
    buffers: Mapping[str, PinnedBuffer],
) -> None:
    _set_buffers(
        module,
        {name: pinned.tensor for name, pinned in buffers.items()},
    )


def _set_buffers(
    module: nn.Module,
    buffers: Mapping[str, torch.Tensor],
) -> None:
    for name, tensor in buffers.items():
        parent, leaf = resolve_parent_leaf(module, name)
        persistent = leaf not in parent._non_persistent_buffers_set
        parent.register_buffer(leaf, tensor, persistent=persistent)


def _named_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    return _unique_name_dict(module.named_parameters(remove_duplicate=False))


def _named_buffers(module: nn.Module) -> dict[str, torch.Tensor]:
    return _unique_name_dict(module.named_buffers(remove_duplicate=False))


def _unique_name_dict(
    items: Iterable[tuple[str, _NamedT]],
) -> dict[str, _NamedT]:
    values: dict[str, _NamedT] = {}
    for name, value in items:
        if name in values:
            raise ValueError(f"Module yielded duplicate name {name!r}.")
        values[name] = value
    return values


def _get_param(parent: nn.Module, leaf: str) -> nn.Parameter:
    param = parent._parameters.get(leaf)
    if param is None:
        raise RuntimeError(f"Parameter {leaf!r} is unexpectedly missing.")
    return param


def _set_param(parent: nn.Module, leaf: str, param: nn.Parameter) -> None:
    if leaf not in parent._parameters:
        raise RuntimeError(f"Parameter {leaf!r} is unexpectedly missing.")
    parent._parameters[leaf] = param


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
    "PinnedBufferTarget",
    "PinnedModuleInstance",
    "PinnedModuleStore",
    "PinnedModuleTarget",
    "PinnedParamTarget",
    "PostCopyHook",
    "PostCopyHookHandle",
]
