"""Composite of offload components, orchestrated as one unit.

A :class:`CompositeComponent` holds an ordered list of offload components
(:class:`~torch_offload.pinned_component.PinnedComponent` /
:class:`~torch_offload.streamed_component.StreamedComponent`) and drives them
together: activate/deactivate as a unit, coordinate the optimizer-step
boundary, aggregate managed names, and route post-copy hooks to the owning
component. It satisfies
:class:`~torch_offload.protocols.ModelStrategyComponent`, so a composite is
itself composable.

The pinned-vs-streamed distinction lives in whoever *builds* the component
list (the store), not here — the composite orchestrates every component
identically. The one consumer that must distinguish them
(:class:`~torch_offload.model_offloader.ModelOffloader`'s checkpointing guard,
which only concerns streamed trainable blocks) filters :attr:`components` by
type.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .pinned_component import PinnedComponent, PinnedComponentStore
from .pinned_module import PostCopyHook, PostCopyHookHandle
from .streamed_component import StreamedComponent, StreamedComponentStore

_OffloadComponent = PinnedComponent | StreamedComponent
_OffloadComponentStore = PinnedComponentStore | StreamedComponentStore


class CompositeComponent:
    """An offload component composed of an ordered list of components.

    Created by binding a :class:`CompositeComponentStore`. Every lifecycle and
    aggregation operation is type-agnostic — the composite never branches on
    whether a component is pinned or streamed.
    """

    def __init__(self, components: Sequence[_OffloadComponent]) -> None:
        self._components = list(components)
        self._teardown_stack: contextlib.ExitStack | None = None

    @property
    def components(self) -> tuple[_OffloadComponent, ...]:
        """The composed components, in activation order."""
        return tuple(self._components)

    @property
    def param_names(self) -> frozenset[str]:
        """Union of every component's managed parameter names."""
        names: set[str] = set()
        for component in self._components:
            names |= component.param_names
        return frozenset(names)

    @property
    def buffer_names(self) -> frozenset[str]:
        """Union of every component's managed buffer names."""
        names: set[str] = set()
        for component in self._components:
            names |= component.buffer_names
        return frozenset(names)

    def component_for_param_name(self, param_name: str) -> _OffloadComponent:
        """The component that manages ``param_name``."""
        for component in self._components:
            if param_name in component.param_names:
                return component
        raise KeyError(
            f"param name {param_name!r} is not managed by this composite"
        )

    def register_post_copy_hook(
        self, param_name: str, hook: PostCopyHook,
    ) -> PostCopyHookHandle:
        """Register a post-copy hook on the component owning ``param_name``."""
        return self.component_for_param_name(param_name).register_post_copy_hook(
            param_name, hook,
        )

    def post_copy_hook_key(self, param_name: str) -> int:
        """Stable hook/dedup key for ``param_name``'s owning component."""
        return self.component_for_param_name(param_name).post_copy_hook_key(
            param_name,
        )

    def activate(self, device: torch.device | str | None = None) -> None:
        """Activate every component, in order, on ``device``.

        Self-cleaning on failure: if a component's ``activate`` raises, the
        already-activated components are deactivated before the exception
        propagates.
        """
        with contextlib.ExitStack() as stack:
            for component in self._components:
                stack.callback(component.deactivate)
                component.activate(device)
            self._teardown_stack = stack.pop_all()

    def deactivate(self) -> None:
        """Deactivate every component. Idempotent — safe before activate or
        multiple times."""
        stack = self._teardown_stack
        self._teardown_stack = None
        if stack is not None:
            stack.close()

    @contextlib.contextmanager
    def optimizer_step(self) -> Iterator[None]:
        """Optimizer-step boundary spanning every component.

        Each component's own ``optimizer_step`` is a guarded no-op when it has
        no active trainables, so entering all of them is uniform and correct.
        """
        with contextlib.ExitStack() as stack:
            for component in self._components:
                stack.enter_context(component.optimizer_step())
            yield


@dataclass(frozen=True, slots=True)
class CompositeComponentStore:
    """Reusable backing stores for a :class:`CompositeComponent`.

    A tuple of component stores (pinned and/or streamed); :meth:`bind` binds
    each to a model and wraps the results in a :class:`CompositeComponent`.
    """

    component_stores: tuple[_OffloadComponentStore, ...]

    def __post_init__(self) -> None:
        if not self.component_stores:
            raise ValueError(
                "CompositeComponentStore requires at least one component store."
            )

    @property
    def cache_bytes(self) -> int:
        return sum(store.cache_bytes for store in self.component_stores)

    @property
    def has_trainables(self) -> bool:
        return any(store.has_trainables for store in self.component_stores)

    def bind(self, model: nn.Module) -> CompositeComponent:
        """Bind every component store to ``model``."""
        return CompositeComponent(
            [store.bind(model) for store in self.component_stores]
        )


__all__ = ["CompositeComponent", "CompositeComponentStore"]
