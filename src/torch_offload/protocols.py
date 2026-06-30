"""Public protocols for cached resources and model bindings.

These Protocols form the contract:

- :class:`ResourceStore` -- cached backing state. A store owns the
  budgeted cache bytes and can be reused across independent activations.

- :class:`ResourceBinding` -- per-use active binding. A binding exposes
  a typed :attr:`value` accessor plus activate/deactivate lifecycle.
  :class:`~torch_offload.model_cache.ModelCache` creates one binding per
  ``use()`` call from a cached store. ``T`` is the type yielded by
  :meth:`~ModelCache.use`.

Top-level :class:`ResourceBinding` implementations in this package:
:class:`~torch_offload.ModelOffloader` (whole-model bulk DMA or streamed
block offload), :class:`~torch_offload.MpsWeights` (whole-model CPU->MPS
materialization without a second CPU cache), and
:class:`~torch_offload.LoRA` (a streamable LoRA factor binding). Its
:class:`ResourceStore` is :class:`~torch_offload.LoRAStore` (pinned LoRA
factor storage).
Future resources (disk-mmap, NVMe-paged, multi-GPU shard) satisfy the
:class:`ResourceStore` / :class:`ResourceBinding` split.

Composable lifecycle pieces inside a model binding include
:class:`~torch_offload.PinnedComponent` and
:class:`~torch_offload.StreamedComponent`.

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
