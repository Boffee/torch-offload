"""Composite of offload components, orchestrated as one unit.

A :class:`CompositeComponent` holds a model's offload components — an
optional pinned remainder
(:class:`~torch_offload.pinned_component.PinnedComponent`) and zero or more
streamed block groups
(:class:`~torch_offload.streamed_component.StreamedComponent`) — and drives
them together: activate/deactivate as a unit, coordinate the optimizer-step
boundary, aggregate managed names, and route post-copy hooks to the owning
component. Like the components it holds, it is a composable
activate/deactivate lifecycle piece, so a composite is itself composable.

The pinned remainder and the streamed groups are stored as separate fields
(the package only ever composes these two kinds), but orchestration is
uniform: every lifecycle and aggregation operation iterates an ordered member
list — pinned first, then the streamed groups — and calls the same method on
each, never branching on concrete type.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from .module_names import buffer_names, parameter_names
from .pinned_component import PinnedComponent, PinnedComponentStore
from .pinned_module import PostCopyHook, PostCopyHookHandle
from .pinned_param import PinnedParam
from .streamed_component import StreamedComponent, StreamedComponentStore


class CompositeComponent:
    """An offload component composed of a pinned remainder and streamed groups.

    Created by binding a :class:`CompositeComponentStore`. The optional pinned
    component and the streamed block groups are kept as separate fields, but
    every lifecycle and aggregation method drives them uniformly through an
    ordered member list (pinned first) — it never branches on whether a member
    is pinned or streamed.
    """

    def __init__(
        self,
        *,
        pinned: PinnedComponent | None,
        streamed: Sequence[StreamedComponent],
    ) -> None:
        self._pinned = pinned
        self._streamed = tuple(streamed)
        self._teardown_stack: contextlib.ExitStack | None = None

    @property
    def pinned(self) -> PinnedComponent | None:
        """The pinned remainder component, or ``None`` when fully streamed."""
        return self._pinned

    @property
    def streamed(self) -> tuple[StreamedComponent, ...]:
        """The streamed block-group components, in activation order."""
        return self._streamed

    def _components(self) -> Iterator[PinnedComponent | StreamedComponent]:
        """Members in activation order: the pinned remainder first (when
        present), then the streamed groups.

        The single definition of member order, derived from ``_pinned`` /
        ``_streamed`` on demand and never cached — so there is no second copy
        that can drift from those two fields.
        """
        if self._pinned is not None:
            yield self._pinned
        yield from self._streamed

    @property
    def param_names(self) -> frozenset[str]:
        """Union of every component's managed parameter names."""
        names: set[str] = set()
        for component in self._components():
            names |= component.param_names
        return frozenset(names)

    @property
    def buffer_names(self) -> frozenset[str]:
        """Union of every component's managed buffer names."""
        names: set[str] = set()
        for component in self._components():
            names |= component.buffer_names
        return frozenset(names)

    def component_for_param_name(
        self, param_name: str,
    ) -> PinnedComponent | StreamedComponent:
        """The component that manages ``param_name``."""
        for component in self._components():
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

    def activate(self, device: torch.device, **kwargs: object) -> None:
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
            for component in self._components():
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
            for component in self._components():
                stack.enter_context(component.optimizer_step())
            yield


@dataclass(frozen=True, slots=True)
class CompositeComponentStore:
    """Reusable backing stores for a :class:`CompositeComponent`.

    Holds the optional pinned remainder store and the streamed block-group
    stores separately; :meth:`bind` binds each to a model and wraps the
    results in a :class:`CompositeComponent`.
    """

    pinned_store: PinnedComponentStore | None
    streamed_stores: tuple[StreamedComponentStore, ...]

    def __post_init__(self) -> None:
        if self.pinned_store is None and not self.streamed_stores:
            raise ValueError(
                "CompositeComponentStore requires a pinned store or at least "
                "one streamed store."
            )

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        blocks_attr: Sequence[str] = (),
        stream_trainable_weights: bool = False,
    ) -> CompositeComponentStore:
        """Decompose ``model`` into a pinned remainder + streamed block groups.

        Each ``blocks_attr`` path becomes one
        :class:`~torch_offload.StreamedComponentStore`; everything those groups
        do not claim becomes a single
        :class:`~torch_offload.PinnedComponentStore`. With no ``blocks_attr``
        (the default), nothing streams — the whole model is one pinned store.
        """
        # One streamed store per block path.
        streamed_stores = tuple(
            StreamedComponentStore.from_module(
                model,
                blocks_path=blocks_path,
                stream_trainable_weights=stream_trainable_weights,
            )
            for blocks_path in blocks_attr
        )

        # The pinned store manages whatever the streamed groups did not claim.
        streamed_params = {n for s in streamed_stores for n in s.param_names}
        streamed_buffers = {n for s in streamed_stores for n in s.buffer_names}
        pinned_params = parameter_names(model) - streamed_params
        pinned_buffers = buffer_names(model) - streamed_buffers
        pinned_store = (
            PinnedComponentStore.from_module(
                model,
                include_param_names=pinned_params,
                include_buffer_names=pinned_buffers,
            )
            if pinned_params or pinned_buffers
            else None
        )

        if pinned_store is None and not streamed_stores:
            raise ValueError(
                "Offloading requires at least one parameter, registered "
                "buffer, or streamed block to manage."
            )
        return cls(pinned_store=pinned_store, streamed_stores=streamed_stores)

    @property
    def cache_bytes(self) -> int:
        pinned = self.pinned_store.cache_bytes if self.pinned_store else 0
        return pinned + sum(s.cache_bytes for s in self.streamed_stores)

    @property
    def has_trainables(self) -> bool:
        pinned = bool(self.pinned_store and self.pinned_store.has_trainables)
        return pinned or any(s.has_trainables for s in self.streamed_stores)

    def pinned_params(self) -> dict[str, PinnedParam]:
        """Every pinned param across pinned + streamed members, by full name."""
        result: dict[str, PinnedParam] = {}
        if self.pinned_store is not None:
            result.update(self.pinned_store.pinned_params())
        for store in self.streamed_stores:
            result.update(store.pinned_params())
        return result

    def bind(
        self, model: nn.Module, *, schedule_model: nn.Module | None = None,
    ) -> CompositeComponent:
        """Bind the pinned and streamed stores to ``model``.

        ``schedule_model`` is forwarded only to the streamed stores — it
        redirects their streaming triggers onto a parallel co-scheduled model
        (see :meth:`~torch_offload.streamed_component.StreamedComponentStore.bind`).
        The pinned store has no streaming and ignores it.
        """
        pinned = self.pinned_store.bind(model) if self.pinned_store else None
        return CompositeComponent(
            pinned=pinned,
            streamed=[
                s.bind(model, schedule_model=schedule_model)
                for s in self.streamed_stores
            ],
        )


__all__ = ["CompositeComponent", "CompositeComponentStore"]
