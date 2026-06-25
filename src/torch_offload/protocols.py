"""Public protocols for cached resources and model bindings.

These Protocols form the contract:

- :class:`ModelStrategyComponent` -- pure lifecycle. A piece composable
  inside a top-level model binding (see :class:`ModelOffloader`).
  Just ``activate(device=...)`` + ``deactivate()``. Components do not
  expose a value because their parent composite owns it.

- :class:`ResourceStore` -- cached backing state. A store owns the
  budgeted cache bytes and can be reused across independent activations.

- :class:`ResourceBinding` -- per-use active binding. A binding exposes
  a typed :attr:`value` accessor plus activate/deactivate lifecycle.
  :class:`~torch_offload.model_cache.ModelCache` creates one binding per
  ``use()`` call from a cached store. ``T`` is the type yielded by
  :meth:`~ModelCache.use`.

- :class:`ModelStrategy` -- model-specific specialization of
  :class:`ResourceBinding[nn.Module]`. Adds a ``model`` convenience
  property for code that works specifically with model bindings.

Top-level :class:`ResourceBinding` implementations in this package:
:class:`~torch_offload.ModelOffloader` (whole-model bulk DMA or streamed
block offload), :class:`~torch_offload.MpsWeights` (whole-model CPU->MPS
materialization without a second CPU cache), and :class:`~torch_offload.LoRA`
(pinned LoRA factor storage).
Future resources (disk-mmap, NVMe-paged, multi-GPU shard) satisfy the
:class:`ResourceStore` / :class:`ResourceBinding` split.

Component implementations include :class:`~torch_offload.PinnedComponent`
and :class:`~torch_offload.StreamedComponent`.

Lifecycle
---------
Backing state is set up before cache admission so store ``cache_bytes``
is final immediately. :class:`~torch_offload.ModelCache` keeps stores
cached and creates bindings for active ``use()`` calls. Stores that
manage trainable parameters may reject concurrent same-key bindings.
The binding lifecycle is then ``activate(device=...)`` (make the value
usable, on the caller-selected device when device-aware) ->
``deactivate()`` (release transient compute resources while store
``cache_bytes`` remains resident).

Package model resources optimize construction peak memory: plain
``torch.Tensor`` parameters may be repointed to pinned storage while
pinning is still in progress. If construction raises after pinning has
started, recovery of the partially constructed model/resource is
unsupported; drop those references and rebuild from a fresh model
instance.

``activate()/deactivate()`` may be repeated as many times as you
want. Device-aware package bindings provide ``use(device)`` for
direct exception-safe use, while :class:`ModelCache` passes the
acquire-time device into ``activate``.

There is no ``close()``. To release store ``cache_bytes`` (typically
pinned host memory), drop the store reference. Python's refcount-based
GC frees pinned tensors immediately. Bindings release what they own on
deactivate; ownership of any user-held model references is the user's
concern.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class ModelStrategyComponent(Protocol):
    """Lifecycle-only contract for a piece composable inside a
    top-level model binding.

    Components do not expose a model -- their parent composite owns
    that. They just contribute to the lifecycle: activate, deactivate.
    The composite calls ``activate(device)`` / ``deactivate()`` on each
    component in order; return values are ignored.
    """

    def activate(
        self, device: torch.device | str | None = None, **kwargs: object,
    ) -> None:
        """Make this piece's contribution ready for compute.

        Implementations may move weights to GPU, allocate a target pool,
        register forward hooks, install an mmap, or do nothing for
        always-resident pieces. Not necessarily re-entrant -- call
        :meth:`deactivate` before activating again. ``device`` lets a
        top-level cache pass through the caller's acquire-time device;
        resources that require an explicit device should raise when it
        is omitted. Extra keyword arguments carry resource-specific
        activation policy (e.g. a streamed component's ``stream_config``);
        resources that don't use them ignore them.
        """
        ...

    def deactivate(self) -> None:
        """Undo :meth:`activate`.

        Should be infallible under normal use: callers may treat a
        raising ``deactivate()`` as unrecoverable for that binding since
        the binding's internal state is unknown after the failure.
        """
        ...


T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class ResourceStore(Protocol):
    """Cached backing state managed by :class:`ModelCache`."""

    @property
    def cache_bytes(self) -> int:
        """Bytes charged against the cache budget."""
        ...


@runtime_checkable
class ResourceBinding(Protocol[T_co]):
    """Per-use cache binding.

    Extends lifecycle methods with a typed :attr:`value` accessor. The
    value must be available immediately after binding; ``activate`` makes
    that value usable for compute.
    """

    @property
    def value(self) -> T_co:
        """The bound payload yielded by :meth:`ModelCache.use`."""
        ...

    def activate(
        self, device: torch.device | str | None = None, **kwargs: object,
    ) -> None:
        """Make the binding ready for compute."""
        ...

    def deactivate(self) -> None:
        """Undo :meth:`activate`. Store bytes remain held."""
        ...


@runtime_checkable
class ModelStrategy(ResourceBinding[nn.Module], Protocol):
    """Model-specific cache binding.

    Adds a ``model`` convenience property (equivalent to :attr:`value`)
    for code that works specifically with model bindings.
    """

    @property
    def model(self) -> nn.Module:
        """The wrapped model for this binding."""
        ...
