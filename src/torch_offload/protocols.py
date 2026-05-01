"""Public protocols for model storage/placement strategies.

Two related Protocols form the contract:

- :class:`ModelStrategyComponent` — pure lifecycle. A piece composable
  inside a top-level strategy (see :class:`ModelOffloader`).
  Just ``cache_bytes`` + ``activate()`` + ``deactivate()``. Components
  don't expose a model because their parent composite owns it.

- :class:`ModelStrategy` — top-level. Extends the component contract
  with model exposure (``model`` property) and context-manager methods
  for the ``with strategy as model:`` pattern. This is what
  :class:`~torch_offload.model_cache.ModelCache` registers and
  manages.

Top-level implementations in this package:
:class:`~torch_offload.PinnedWeights` (whole-model bulk DMA between
pinned CPU and GPU) and :class:`ModelOffloader` (composite of
streamers + pinning + trainable handling). Future strategies (disk-mmap,
NVMe-paged, multi-GPU shard) just satisfy :class:`ModelStrategy`.

Component implementations: :class:`~torch_offload.StreamedWeights`,
:class:`~torch_offload.TrainableWeights`. (And :class:`PinnedWeights`
also satisfies the component shape — composites use it inline.)

Lifecycle
---------
``__init__`` sets up backing storage (pinning, etc.) so
``cache_bytes`` is final immediately and a top-level strategy is ready
for :class:`~torch_offload.model_cache.ModelCache` admission →
``activate()`` (make model usable) → ``deactivate()`` (release transient
compute resources, keep ``cache_bytes`` resident).

``activate()/deactivate()`` may be repeated as many times as you
want. Top-level strategies are also context managers:
``with strategy as model: ...`` is equivalent to ``activate()`` then
yielding ``strategy.model``.

There is no ``close()``. To release ``cache_bytes`` (typically
pinned host memory), drop the strategy reference (and the model
reference if you don't need it anymore). Python's refcount-based
GC frees pinned tensors immediately. Strategies release what they
own; ownership of the user's model is the user's concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Protocol, runtime_checkable

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
    activate, deactivate. The composite calls ``activate()`` /
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

    def activate(self) -> None:
        """Make this piece's contribution ready for compute.

        Implementations may move weights to GPU, allocate a slot pool,
        register forward hooks, install an mmap, or do nothing for
        always-resident pieces. Not necessarily re-entrant — call
        :meth:`deactivate` before activating again.
        """
        ...

    def deactivate(self) -> None:
        """Undo :meth:`activate`. ``cache_bytes`` remains held.

        Should be infallible under normal use: the cache treats a
        raising ``deactivate()`` as a poisoned strategy and drops it
        (without further cleanup attempts) since the strategy's
        internal state is unknown after the failure. After deactivate,
        the caller drops the strategy reference to release pinned
        memory — there is no separate ``close()`` step.
        """
        ...


@runtime_checkable
class ModelStrategy(ModelStrategyComponent, Protocol):
    """Top-level strategy: a registerable, model-exposing variant of
    :class:`ModelStrategyComponent`.

    Adds ``model`` (so callers can reach the wrapped module) plus
    context-manager methods (so ``with strategy as model: ...`` works).
    This is what :class:`~torch_offload.model_cache.ModelCache`
    registers and manages.

    The ``@runtime_checkable`` decoration enables ``isinstance(x,
    ModelStrategy)`` for sanity checks, but the check is structural
    and only verifies attribute *presence*, not signatures or types.
    Treat it as a weak guard, not a contract verifier.
    """

    @property
    def model(self) -> nn.Module:
        """The wrapped model. Stable across activate/deactivate cycles
        (the same Module is returned regardless of whether weights are
        currently GPU-resident or pinned-CPU).

        Must be available immediately after construction and must not
        depend on activation state. ``ModelCache`` reads this getter
        BEFORE calling :meth:`activate` to avoid a post-activation
        exception window where a raising getter would skip the
        deactivate path on an already-active strategy."""
        ...

    def __enter__(self) -> nn.Module:
        """Calls :meth:`activate`, returns :attr:`model`."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        """Equivalent to :meth:`deactivate`."""
        ...
