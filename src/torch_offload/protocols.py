"""Public protocols for cached stores and runtime bindings.

These Protocols form the contract:

- :class:`ResourceSpec` -- lazy cache registration. A spec identifies one
  resource, estimates its backing bytes, builds its store, and projects a
  leased store to the value exposed to its caller.

- :class:`ResourceStore` -- cached backing state. A store owns the
  budgeted cache bytes and defines whether active uses may overlap.

- :class:`ResourceBinding` -- active-resource lifecycle. A binding exposes a
  typed :attr:`value` accessor plus activate/deactivate methods. A cached
  resource may implement this protocol directly, as :class:`ModelOffloader`
  does. The cache does not know about this protocol.

Top-level :class:`ResourceBinding` implementations in this package:
:class:`~torch_offload.ModelOffloader` (whole-model bulk DMA or streamed
block offload) and :class:`~torch_offload.MpsWeights` (whole-model
CPU->MPS materialization without a second CPU cache). A
:class:`~torch_offload.LoRA` is itself the cached adapter resource. Merge
consumers read its immutable pinned backing directly; its exclusive routed
lifecycle is driven by the :class:`~torch_offload.ModelOffloader` it is
attached to.

Composable lifecycle pieces inside a model runtime include
:class:`~torch_offload.PinnedComponent` and
:class:`~torch_offload.StreamedComponent`.

Lifecycle
---------
Backing state is set up before cache admission so store ``cache_bytes``
is final immediately. :class:`~torch_offload.ResourceCache` keeps stores
cached and protects them with reference-counted leases. A store may itself
implement the active lifecycle. That lifecycle is
``activate(device=...)`` (make the value usable, on the caller-selected
device when device-aware) ->
``deactivate()`` (release transient compute resources while store
``cache_bytes`` remains resident).

Package model resources optimize construction peak memory: plain
``torch.Tensor`` parameters may be repointed to pinned storage while
pinning is still in progress. If construction raises after pinning has
started, recovery of the partially constructed model/resource is
unsupported; drop those references and rebuild from a fresh model
instance.

``activate()/deactivate()`` may be repeated as many times as you want.
:class:`~torch_offload.CachedModelRunner` combines cache leases with an
exception-safe model activation scope.

There is no ``close()``. To release store ``cache_bytes`` (typically
pinned host memory), drop the store reference. Python's refcount-based
GC frees pinned tensors immediately. Bindings release what they own on
deactivate; ownership of any user-held model references is the user's
concern.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

import torch

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class ResourceStore(Protocol):
    """Cached backing state managed by :class:`ResourceCache`."""

    @property
    def cache_bytes(self) -> int:
        """Bytes charged against the cache budget."""
        ...


class ResourceSpec(Protocol[T_co]):
    """Structural contract for one lazily built cache entry.

    ``key`` is the cache identity and must include every construction input
    that affects the resulting resource. ``estimated_cache_bytes`` is used
    for pre-build admission; the cache reconciles it with the built store's
    actual :attr:`ResourceStore.cache_bytes`.
    """

    @property
    def key(self) -> str:
        """Caller-chosen cache identity."""
        ...

    @property
    def estimated_cache_bytes(self) -> int:
        """Pre-build estimate used for admission planning."""
        ...

    def build_store(self) -> ResourceStore:
        """Build fresh reusable backing storage on a cache miss."""
        ...

    def value(self, store: ResourceStore) -> T_co:
        """Project the leased store to the value returned to the caller."""
        ...


@runtime_checkable
class ResourceBinding(Protocol[T_co]):
    """Active-resource lifecycle.

    Extends lifecycle methods with a typed :attr:`value` accessor. The
    value is available before activation; ``activate`` makes it usable for
    compute.
    """

    @property
    def value(self) -> T_co:
        """The bound payload made usable by :meth:`activate`."""
        ...

    def activate(
        self,
        device: torch.device | str | None = None,
        **kwargs: object,
    ) -> None:
        """Make the binding ready for compute."""
        ...

    def deactivate(self) -> None:
        """Undo :meth:`activate`. Store bytes remain held."""
        ...
