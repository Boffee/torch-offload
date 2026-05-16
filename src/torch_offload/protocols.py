"""Public protocols for cached resources and model strategies.

Three Protocols form the contract:

- :class:`ModelStrategyComponent` — pure lifecycle. A piece composable
  inside a top-level strategy (see :class:`ModelOffloader`).
  Just ``cache_bytes`` + ``activate(device=...)`` + ``deactivate()``.
  Components don't expose a value because their parent composite owns it.

- :class:`CachedResource` — generic top-level cache contract.
  Extends the component lifecycle with a typed :attr:`value` accessor.
  This is what
  :class:`~torch_offload.model_cache.ModelCache` registers and
  manages.  ``T`` is the type yielded by :meth:`~ModelCache.use`.

- :class:`ModelStrategy` — model-specific specialization of
  :class:`CachedResource[nn.Module]`.  Adds a ``model`` convenience
  property for code that works specifically with model strategies.

Top-level :class:`CachedResource` implementations in this package:
:class:`~torch_offload.PinnedWeights` (whole-model bulk DMA between
pinned CPU and GPU), :class:`ModelOffloader` (composite of
streamers + pinning + trainable handling), and :class:`~torch_offload.LoRA`
(pinned LoRA factor storage). Future resources (disk-mmap, NVMe-paged,
multi-GPU shard) just satisfy :class:`CachedResource`.

Component implementations: :class:`~torch_offload.StreamedWeights`,
:class:`~torch_offload.TrainableWeights`. (And :class:`PinnedWeights`
also satisfies the component shape — composites use it inline.)

Lifecycle
---------
``__init__`` sets up backing storage (pinning, etc.) so
``cache_bytes`` is final immediately and a top-level resource is ready
for :class:`~torch_offload.model_cache.ModelCache` admission →
``activate(device=...)`` (make resource usable, on the caller-selected
device when the resource has device placement) → ``deactivate()``
(release transient compute resources, keep ``cache_bytes`` resident).
Package strategies optimize construction peak memory: plain
``torch.Tensor`` parameters may be repointed to pinned storage while
pinning is still in progress. If construction raises after pinning has
started, recovery of the partially constructed model/resource is
unsupported; drop those references and rebuild from a fresh model
instance.

``activate()/deactivate()`` may be repeated as many times as you
want. Device-aware package strategies provide ``use(device)`` for
direct exception-safe use, while :class:`ModelCache` passes the
acquire-time device into ``activate``.

There is no ``close()``. To release ``cache_bytes`` (typically
pinned host memory), drop the resource reference. Python's refcount-based
GC frees pinned tensors immediately. Resources release what they
own; ownership of the user's model is the user's concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar, runtime_checkable

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class SlotOwnership:
    """Identifies a parameter or buffer slot in the model tree by
    ``(parent_module, leaf_name, kind)``.

    Used as a slot-skip filter when one strategy manages a subset of
    a model's slots and a second strategy needs to ignore them. Unlike
    ``id(param)`` / ``id(buffer)``, this identity survives
    ``module._parameters[leaf] = new_param`` swaps — the parent module
    and leaf name are stable even when the Parameter/buffer object at
    that slot changes. That decouples filter consumers from
    construction order: a strategy can be built with the filter at any
    time, before or after the producing strategy has mutated slots.

    ``parent_id`` is ``id()`` of the parent module, which is stable for
    the module's lifetime and unique per submodule (Python guarantee
    while a reference is held).
    """

    parent_id: int
    leaf: str
    kind: Literal["param", "buffer"]


@runtime_checkable
class ModelStrategyComponent(Protocol):
    """Lifecycle-only contract for a piece composable inside a
    top-level strategy.

    Components don't expose a model — their parent composite owns
    that. They just contribute to the lifecycle: report cache budget,
    activate, deactivate. The composite calls ``activate(device)`` /
    ``deactivate()`` on each component in order; return values are
    ignored.

    A top-level :class:`ModelStrategy` is also a component (it extends
    this Protocol), so a strategy can be used standalone OR as a piece
    of a larger composite.
    """

    @property
    def cache_bytes(self) -> int:
        """Bytes charged against ``ModelCache.max_cache_bytes``.

        Typically pinned host memory, but a strategy may report any
        resource it wants the cache to budget against — staging buffers,
        mmap regions, etc. Components that don't consume cache budget
        (e.g. :class:`TrainableWeights` which only nudges existing
        params) should return 0.
        """
        ...

    def activate(self, device: torch.device | str | None = None) -> None:
        """Make this piece's contribution ready for compute.

        Implementations may move weights to GPU, allocate a slot pool,
        register forward hooks, install an mmap, or do nothing for
        always-resident pieces. Not necessarily re-entrant — call
        :meth:`deactivate` before activating again. ``device`` lets a
        top-level cache choose placement at acquire time; resources that
        require placement should raise when it is omitted.
        """
        ...

    def deactivate(self) -> None:
        """Undo :meth:`activate`. ``cache_bytes`` remains held.

        Should be infallible under normal use: the cache treats a
        raising ``deactivate()`` as unrecoverable and drops the strategy
        (without further cleanup attempts) since the strategy's internal
        state is unknown after the failure. After deactivate, the caller
        drops the strategy reference to release pinned memory — there is
        no separate ``close()`` step.
        """
        ...


T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class CachedResource(Protocol[T_co]):
    """Top-level cache-managed resource.

    Extends the component lifecycle with a typed :attr:`value`
    accessor.  This is the plug-in contract that
    :class:`~torch_offload.model_cache.ModelCache` consumes.

    ``T`` is the type yielded by :meth:`~ModelCache.use`:
    ``nn.Module`` for model strategies, ``LoRA`` for standalone
    LoRA factor storage, etc.
    """

    @property
    def cache_bytes(self) -> int:
        """Bytes charged against the cache budget."""
        ...

    @property
    def value(self) -> T_co:
        """The cached payload, yielded by :meth:`~ModelCache.use`.

        Must be available immediately after construction and must not
        depend on activation state.
        """
        ...

    def activate(self, device: torch.device | str | None = None) -> None:
        """Make the resource ready for compute.

        ``device`` is optional at the protocol boundary so device-neutral
        resources can ignore it. Device-aware resources should require an
        explicit placement and raise when it is omitted.
        """
        ...

    def deactivate(self) -> None:
        """Undo :meth:`activate`.  ``cache_bytes`` remains held."""
        ...

@runtime_checkable
class ModelStrategy(CachedResource[nn.Module], Protocol):
    """Model-specific cached resource.

    Adds a ``model`` convenience property (equivalent to :attr:`value`)
    for code that works specifically with model strategies.
    """

    @property
    def model(self) -> nn.Module:
        """The wrapped model.  Stable across activate/deactivate cycles."""
        ...
