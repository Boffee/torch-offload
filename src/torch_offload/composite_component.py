"""Composite of offload components, orchestrated as one unit.

A :class:`CompositeComponent` holds an ordered list of offload components
(:class:`~torch_offload.pinned_component.PinnedComponent` /
:class:`~torch_offload.streamed_component.StreamedComponent`) and drives them
together: activate/deactivate as a unit, coordinate the optimizer-step
boundary, aggregate managed names, and route post-copy hooks to the owning
component. It satisfies
:class:`~torch_offload.protocols.ModelStrategyComponent`, so a composite is
itself composable.

The pinned-vs-streamed distinction lives in the builder
(:meth:`CompositeComponentStore.from_module`), not in the orchestration:
:class:`CompositeComponent` drives every component identically through the
:class:`~torch_offload.protocols.OffloadComponent` protocol, never branching
on concrete type.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .module_names import buffer_names, parameter_names
from .pinned_component import PinnedComponentStore
from .pinned_module import PostCopyHook, PostCopyHookHandle
from .protocols import OffloadComponent, OffloadComponentStore
from .streamed_component import StreamedComponentStore


class CompositeComponent:
    """An offload component composed of an ordered list of components.

    Created by binding a :class:`CompositeComponentStore`. Every lifecycle and
    aggregation operation is type-agnostic — the composite never branches on
    whether a component is pinned or streamed.
    """

    def __init__(self, components: Sequence[OffloadComponent]) -> None:
        self._components = list(components)
        self._teardown_stack: contextlib.ExitStack | None = None

    @property
    def components(self) -> tuple[OffloadComponent, ...]:
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

    def component_for_param_name(self, param_name: str) -> OffloadComponent:
        """The component that manages ``param_name``."""
        for component in self._components:
            if param_name in component.param_names:
                return component
        raise KeyError(
            f"param name {param_name!r} is not managed by this composite"
        )

    def register_post_copy_hook(
        self, name: str, hook: PostCopyHook,
    ) -> PostCopyHookHandle:
        """Register a post-copy hook on the component owning ``name``."""
        return self.component_for_param_name(name).register_post_copy_hook(
            name, hook,
        )

    def activate(
        self, device: torch.device | str | None = None, **kwargs: object,
    ) -> None:
        """Activate every component, in order, on ``device``.

        Self-cleaning on failure: if a component's ``activate`` raises, the
        already-activated components are deactivated before the exception
        propagates. Extra keyword arguments (e.g. a streamed member's
        ``stream_config``) are forwarded to every component; members that
        don't use them ignore them.
        """
        if self._teardown_stack is not None:
            raise RuntimeError(
                "CompositeComponent.activate() called while already active; "
                "deactivate() first."
            )
        with contextlib.ExitStack() as stack:
            for component in self._components:
                stack.callback(component.deactivate)
                component.activate(device, **kwargs)
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

    component_stores: tuple[OffloadComponentStore, ...]

    def __post_init__(self) -> None:
        if not self.component_stores:
            raise ValueError(
                "CompositeComponentStore requires at least one component store."
            )

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        blocks_attr: list[str] = [],  # noqa: B006  (read-only; never mutated)
        stream_trainable_weights: bool = False,
    ) -> CompositeComponentStore:
        """Decompose ``model`` into a pinned remainder + streamed block groups.

        Each ``blocks_attr`` path becomes one
        :class:`~torch_offload.StreamedComponentStore`; everything those groups
        do not claim becomes a single
        :class:`~torch_offload.PinnedComponentStore`, placed first so it
        activates before the streamed groups. With no ``blocks_attr`` (the
        default), nothing streams — the whole model is one pinned component.
        """
        # One streamed component per block path.
        streamed_stores = tuple(
            StreamedComponentStore.from_module(
                model,
                blocks_path=blocks_path,
                stream_trainable_weights=stream_trainable_weights,
            )
            for blocks_path in blocks_attr
        )

        # The pinned component manages whatever the streamed groups did not claim.
        streamed_params = {n for s in streamed_stores for n in s.param_names}
        streamed_buffers = {n for s in streamed_stores for n in s.buffer_names}
        pinned_params = parameter_names(model) - streamed_params
        pinned_buffers = buffer_names(model) - streamed_buffers

        # Pinned first (preserves activation order), then the streamed groups.
        stores: list[PinnedComponentStore | StreamedComponentStore] = []
        if pinned_params or pinned_buffers:
            stores.append(
                PinnedComponentStore.from_module(
                    model,
                    include_param_names=pinned_params,
                    include_buffer_names=pinned_buffers,
                )
            )
        stores.extend(streamed_stores)
        if not stores:
            raise ValueError(
                "Offloading requires at least one parameter, registered "
                "buffer, or streamed block to manage."
            )
        return cls(tuple(stores))

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
