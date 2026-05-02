"""LRU resource cache over :class:`CachedResource` plug-ins.

Manages cached pinned-CPU (or other resource-owned) backing storage for
multiple independent resources (models, LoRA adapters, etc.). When a new
resource needs more cache than is free, the least-recently-used inactive
entries are evicted until room is available. Active entries (currently
inside a ``use()`` context) are never evicted.

Design highlights
-----------------
- **Resource-agnostic.** The cache only talks to the
  :class:`CachedResource` protocol — lifecycle methods plus
  ``cache_bytes`` accounting. Pluggable: today :class:`PinnedWeights`,
  :class:`ModelOffloader`, and :class:`LoRA`; future resources
  (disk-mmap, NVMe-paged, multi-GPU shard) just satisfy the protocol.
- **Active-set with refcount.** Multiple keys can be active
  simultaneously (e.g. text encoder and embedding processor in the
  same call), and the same key can be acquired re-entrantly (refcount
  bump, the underlying strategy is not re-activated).
- **Transactional admission, with one caveat.** Activation failures
  drop the poisoned entry and propagate :class:`ActivationError`.
  Eviction is just a reference drop (no failure path) — pinned memory
  is freed when GC runs on the dropped strategy. **Factory failure**
  preserves the registration but leaves any pre-eviction *committed*
  — the cache evicts inactive LRU entries to give the factory a
  predictable host-memory budget for pinning, and those evictions
  are not rolled back if the factory raises (rolling back would
  mean re-pinning the just-released weights, which can OOM the host
  allocator). The cache stays internally consistent; the cost is
  some warm cached entries disappearing.
- **No GPU budget.** The cache only enforces ``max_cache_bytes``
  (typically pinned host memory). Concurrent active models share the
  GPU at the caller's risk.
- **Single-thread.** No locking. Sequential callers only.

Instance-owned (not global) so it's library-friendly and embeddable.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, overload

import torch

from .protocols import CachedResource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


T = TypeVar("T")


@dataclass(frozen=True)
class ResourceSpec(Generic[T]):
    """Registration spec for a cached resource.

    ``key`` is the cache identity — caller-chosen and must include any
    construction inputs that affect the resulting weights or structure
    (checkpoint hash, dtype, quantization config, LoRA stack, etc.). The
    cache does not introspect the factory; calling ``use()`` with the
    same key but a different factory silently returns the cached entry.

    ``estimated_cache_bytes`` is used to evict before building. The
    cache reconciles against the actual ``resource.cache_bytes`` after
    construction; if the actual exceeds the estimate enough to overflow
    the budget, the cache evicts more or rejects the admission.

    ``label`` is an optional human-readable name surfaced in logs and
    snapshots; defaults to ``None`` if omitted (in which case displays
    fall back to ``key``).

    .. note::
       The ``factory`` should build a *fresh* resource that the cache
       solely owns. Eviction releases backing storage only when
       the resource becomes unreachable — Python's refcount-based GC
       handles this when the cache is the sole owner.
    """

    key: str
    estimated_cache_bytes: int
    factory: Callable[[], CachedResource[T]]
    label: str | None = None


ModelSpec: type[ResourceSpec] = ResourceSpec


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Per-key snapshot returned by :meth:`ModelCache.info`."""

    key: str
    label: str | None
    estimated_cache_bytes: int
    cache_bytes: int | None  # None when not built (registered but never used)
    cached: bool
    active_count: int


@dataclass
class ModelCacheStats:
    """Mutable counters tracking cache activity over its lifetime."""

    hits: int = 0
    misses: int = 0
    builds: int = 0
    evictions: int = 0
    bytes_evicted: int = 0
    factory_errors: int = 0
    activation_errors: int = 0
    peak_cache_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ModelCacheSnapshot:
    """Detached point-in-time view of the cache.

    The dataclass itself is frozen, ``active_refcounts`` is a tuple of
    ``(key, count)`` pairs, and ``stats`` is deep-copied at capture
    time so subsequent cache activity does not mutate it. The nested
    ``ModelCacheStats`` instance is not additionally frozen for
    pragmatic reasons (the cache's working stats need to be mutable),
    so callers should not mutate it.
    """

    max_cache_bytes: int
    used_cache_bytes: int
    registered_keys: tuple[str, ...]
    cached_keys_lru_to_mru: tuple[str, ...]
    active_refcounts: tuple[tuple[str, int], ...]
    stats: ModelCacheStats


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ModelCacheError(RuntimeError):
    """Base for all cache errors."""


class ModelNotRegisteredError(ModelCacheError):
    """``use(str)`` called for a key that has no registration."""


class DuplicateModelKeyError(ModelCacheError):
    """``register(spec)`` called for a key that is already registered
    (without ``replace=True``)."""


class ModelTooLargeError(ModelCacheError):
    """Even after evicting all inactive entries, the requested model
    cannot fit in the budget. Carries enough context to debug."""

    def __init__(
        self,
        *,
        key: str,
        required: int,
        used: int,
        limit: int,
        active_refcounts: dict[str, int],
    ) -> None:
        short = used + required - limit
        super().__init__(
            f"Cannot admit {key!r}: needs {required} bytes, "
            f"{used}/{limit} already used (short by {short}). "
            f"Active non-evictable entries: {active_refcounts}"
        )
        self.key = key
        self.required = required
        self.used = used
        self.limit = limit
        self.active_refcounts = active_refcounts


class ModelInUseError(ModelCacheError):
    """``evict(key)`` or ``clear()`` called while one or more entries
    are active."""


class ActivationError(ModelCacheError):
    """A strategy's ``activate()`` raised. The cache discards the entry
    (drops the strategy reference, removes it from cache state) regardless
    of whether the entry was freshly built or previously cached —
    strategies with multi-step ``activate()`` (e.g.
    :func:`ModelOffloader`) can fail mid-way after partially
    installing hooks/pool/composed PinnedWeights, and caching such an
    entry as "ready to retry" lies about its state. The next acquire
    rebuilds via the registered factory."""


# ---------------------------------------------------------------------------
# Internal entry state
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    spec: ResourceSpec
    strategy: CachedResource | None = None
    cache_bytes: int = 0           # actual, post-build
    active_count: int = 0
    active_value: Any = None  # cached for re-entrant acquire


# ---------------------------------------------------------------------------
# ModelCache
# ---------------------------------------------------------------------------


class ModelCache:
    """LRU pool over :class:`CachedResource` instances.

    Parameters
    ----------
    max_cache_bytes:
        Total budget for cached strategy backing storage (typically
        pinned host memory). Must be ≥ 0.
    empty_host_cache:
        Optional callback invoked after every successful eviction (and
        after dropping a rejected newly-built strategy) to flush PyTorch's
        ``CachingHostAllocator`` so freed pinned pages return to the OS.
        If ``None`` and CUDA is available, defaults to
        ``torch._C._host_emptyCache`` when present. Pass a no-op
        callable to disable.
    """

    def __init__(
        self,
        max_cache_bytes: int,
        *,
        empty_host_cache: Callable[[], None] | None = None,
    ) -> None:
        if max_cache_bytes < 0:
            raise ValueError(f"max_cache_bytes must be >= 0, got {max_cache_bytes}")
        self._max_cache_bytes = max_cache_bytes
        self._empty_host_cache = self._resolve_host_cache_cb(empty_host_cache)
        # Insertion order is meaningful for snapshots only; LRU is tracked
        # separately so eviction order doesn't depend on registration order.
        self._entries: dict[str, _Entry] = {}
        # Inactive entries only. MRU at the right end.
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._used_bytes = 0
        self._stats = ModelCacheStats()

    @staticmethod
    def _resolve_host_cache_cb(cb: Callable[[], None] | None) -> Callable[[], None] | None:
        if cb is not None:
            return cb
        # Best-effort default: PyTorch's CachingHostAllocator flush is
        # only present when CUDA is available and only on torch >= 2.x.
        host_empty: Any = getattr(torch._C, "_host_emptyCache", None)
        if callable(host_empty) and torch.cuda.is_available():
            return host_empty
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def max_cache_bytes(self) -> int:
        return self._max_cache_bytes

    @property
    def used_cache_bytes(self) -> int:
        return self._used_bytes

    def register(self, spec: ResourceSpec, *, replace: bool = False) -> None:
        """Register a lazy model factory without building it.

        Idempotent only with ``replace=True``: re-registering an
        existing key without ``replace`` raises
        :class:`DuplicateModelKeyError`. With ``replace=True``, any
        existing cached entry is evicted first (which raises
        :class:`ModelInUseError` if the key is active).
        """
        if spec.estimated_cache_bytes < 0:
            raise ValueError(
                f"spec.estimated_cache_bytes must be >= 0, got {spec.estimated_cache_bytes}"
            )
        if spec.key in self._entries:
            if not replace:
                raise DuplicateModelKeyError(f"{spec.key!r} is already registered")
            self._evict_for_replace(spec.key)
        self._entries[spec.key] = _Entry(spec=spec)

    def unregister(self, key: str, *, evict: bool = True) -> None:
        """Drop a registration. If a built strategy exists and
        ``evict=True`` (default), evict it; otherwise raise
        :class:`ModelInUseError` if it's cached or active."""
        entry = self._entries.get(key)
        if entry is None:
            return
        if entry.active_count > 0:
            raise ModelInUseError(
                f"cannot unregister active model {key!r} (refcount={entry.active_count})"
            )
        if entry.strategy is not None:
            if not evict:
                raise ModelInUseError(
                    f"{key!r} has a built strategy; pass evict=True to release it"
                )
            self._evict_inactive(key)
        del self._entries[key]

    @overload
    def use(self, model: ResourceSpec[T]) -> contextlib.AbstractContextManager[T]: ...
    @overload
    def use(self, model: str) -> contextlib.AbstractContextManager[Any]: ...

    @contextlib.contextmanager
    def use(self, model: str | ResourceSpec) -> Iterator[Any]:
        """Acquire an active lease on a cached resource.

        Accepts either a registered key (string) or a :class:`ResourceSpec`
        (auto-registers if its key isn't already known). Yields the
        resource's :attr:`~CachedResource.value` for use; on context
        exit, releases the lease.

        Re-entrant for the same key: nested ``use()`` calls bump a
        per-key refcount and share the already-active value (the
        underlying resource is *not* activated again).
        """
        spec = model if isinstance(model, ResourceSpec) else None
        key = spec.key if spec is not None else model
        assert isinstance(key, str)

        entry = self._entries.get(key)
        if entry is None:
            if spec is None:
                raise ModelNotRegisteredError(
                    f"{key!r} is not registered; pass a ResourceSpec to use() or call register() first"
                )
            self.register(spec)
            entry = self._entries[key]

        with self._activate_entry(entry):
            assert entry.active_value is not None
            yield entry.active_value

    def evict(self, key: str) -> None:
        """Manually evict one inactive cached entry. Raises
        :class:`ModelInUseError` if active, no-op if not cached."""
        entry = self._entries.get(key)
        if entry is None or entry.strategy is None:
            return
        if entry.active_count > 0:
            raise ModelInUseError(f"cannot evict active model {key!r}")
        self._evict_inactive(key)

    def clear(self) -> None:
        """Evict all inactive entries. Registrations are preserved.
        Raises :class:`ModelInUseError` if any entry is active."""
        active = [k for k, e in self._entries.items() if e.active_count > 0]
        if active:
            raise ModelInUseError(f"cannot clear while models are active: {active}")
        for key in list(self._lru):
            self._evict_inactive(key)

    def info(self, key: str) -> ModelInfo:
        """Per-key snapshot. Raises :class:`ModelNotRegisteredError` if
        unknown."""
        entry = self._entries.get(key)
        if entry is None:
            raise ModelNotRegisteredError(f"{key!r} is not registered")
        return ModelInfo(
            key=entry.spec.key,
            label=entry.spec.label,
            estimated_cache_bytes=entry.spec.estimated_cache_bytes,
            cache_bytes=entry.cache_bytes if entry.strategy is not None else None,
            cached=entry.strategy is not None,
            active_count=entry.active_count,
        )

    def snapshot(self) -> ModelCacheSnapshot:
        """Whole-cache point-in-time view. ``stats`` is copied at
        capture time so subsequent activity does not mutate the
        snapshot; the dataclass itself is frozen and
        ``active_refcounts`` is a tuple. See
        :class:`ModelCacheSnapshot` for the immutability nuances."""
        return ModelCacheSnapshot(
            max_cache_bytes=self._max_cache_bytes,
            used_cache_bytes=self._used_bytes,
            registered_keys=tuple(self._entries.keys()),
            cached_keys_lru_to_mru=tuple(self._lru.keys()),
            active_refcounts=tuple(
                (k, e.active_count)
                for k, e in self._entries.items()
                if e.active_count > 0
            ),
            stats=dataclasses.replace(self._stats),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _activate_entry(self, entry: _Entry) -> Iterator[None]:
        key = entry.spec.key
        # Re-entrant case: already active for this key. Bump refcount,
        # share the active module, do not re-activate the strategy.
        if entry.active_count > 0:
            entry.active_count += 1
            try:
                yield
            finally:
                entry.active_count -= 1
            return

        # First lease: ensure built, then activate.
        if entry.strategy is None:
            self._build_into_entry(entry)
        else:
            self._stats.hits += 1
            self._lru.pop(key, None)  # leaving inactive set

        assert entry.strategy is not None
        # Fetch value BEFORE activate. The Protocol contract says
        # `strategy.value` is stable across cycles (available regardless
        # of activation state), and reading first eliminates a post-
        # activate exception window where a raising `value` getter
        # would skip the deactivate path on the now-active strategy.
        value = entry.strategy.value
        try:
            entry.strategy.activate()
        except BaseException as exc:
            self._stats.activation_errors += 1
            # Treat all activation failures as poisoned regardless of
            # whether the entry was freshly built or previously cached.
            # Block-streaming strategies can fail mid-way through
            # activate after partially registering hooks / allocating
            # GPU pool / activating composed PinnedWeights. Caching
            # such an entry as "ready to retry" lies about its state
            # and the next acquire would operate on partially-installed
            # resources.
            self._discard_entry(entry)
            raise ActivationError(f"activate() failed for {key!r}") from exc

        # Reconcile cache_bytes after a successful activate. Most
        # strategies pin in __init__ so cache_bytes is final at
        # admission, but a strategy that defers pinning (e.g., a custom
        # one) might report 0 at admission and pin during activate.
        # Without this update _used_bytes would lag reality and future
        # admissions would over-commit. If the actual now exceeds
        # max_cache_bytes we can't unwind mid-context (the strategy is
        # active and being
        # yielded), so we log and continue — the user is over budget
        # but at least the accounting reflects it.
        post_activate_bytes = entry.strategy.cache_bytes
        if post_activate_bytes < 0:
            # Same guard as the factory-admission path. A misbehaving
            # strategy returning negative cache_bytes after activate
            # would corrupt _used_bytes accounting just as it would
            # at admission. Treat the activation as failed and discard.
            with contextlib.suppress(BaseException):
                entry.strategy.deactivate()
            self._stats.activation_errors += 1
            self._discard_entry(entry)
            raise ModelCacheError(
                f"strategy.cache_bytes for {key!r} returned "
                f"{post_activate_bytes} (must be >= 0) after activate"
            )
        if post_activate_bytes != entry.cache_bytes:
            delta = post_activate_bytes - entry.cache_bytes
            entry.cache_bytes = post_activate_bytes
            self._used_bytes += delta
            self._stats.peak_cache_bytes = max(
                self._stats.peak_cache_bytes, self._used_bytes
            )
            if self._used_bytes > self._max_cache_bytes:
                logger.warning(
                    "ModelCache over budget after activate(): %r grew from "
                    "%d to %d bytes; total %d/%d. The strategy reported a "
                    "smaller cache_bytes at construction than after "
                    "activate() — usually means a custom strategy that "
                    "defers pinning. Pin in __init__ so cache_bytes is "
                    "final at admission.",
                    key, entry.cache_bytes - delta, post_activate_bytes,
                    self._used_bytes, self._max_cache_bytes,
                )

        entry.active_value = value
        entry.active_count = 1
        try:
            yield
        finally:
            entry.active_count -= 1
            if entry.active_count == 0:
                # Strategy contract: deactivate is expected to leave
                # the strategy in a clean inactive state and not raise
                # under normal use. If it does raise, the strategy's
                # internal state is unknown — don't risk reusing it.
                # Discard so a subsequent acquire rebuilds.
                try:
                    entry.strategy.deactivate()
                except BaseException:
                    self._discard_entry(entry, was_active=True)
                    raise
                entry.active_value = None
                # Active → inactive: re-enter LRU at MRU position.
                self._lru[key] = None

    def _build_into_entry(self, entry: _Entry) -> None:
        """Cache miss: evict to make room, build, reconcile actuals."""
        key = entry.spec.key
        self._stats.misses += 1
        estimate = entry.spec.estimated_cache_bytes
        # Pre-build eviction: trust the estimate.
        self._evict_until_room(key, estimate)
        try:
            strategy = entry.spec.factory()
        except BaseException:
            self._stats.factory_errors += 1
            raise
        actual = strategy.cache_bytes
        if actual < 0:
            # Misbehaving strategy. Don't admit it (would corrupt
            # _used_bytes accounting). Drop the local ref BEFORE the
            # host-cache flush so refcount-GC frees the strategy's
            # pinned tensors in time for empty_host_cache to actually
            # reclaim them.
            del strategy
            self._after_release()
            raise ModelCacheError(
                f"strategy.cache_bytes for {key!r} returned {actual} (must be >= 0)"
            )
        # Reconcile if actual overshot the estimate.
        if actual > estimate:
            try:
                self._evict_until_room(key, actual)
            except BaseException:
                # Can't fit the actual size — drop the new strategy and
                # re-raise. Don't mark as built. del-before-flush so
                # refcount-GC frees the strategy before the host-cache
                # flush runs.
                del strategy
                self._after_release()
                raise
        entry.strategy = strategy
        entry.cache_bytes = actual
        self._used_bytes += actual
        self._stats.builds += 1
        self._stats.peak_cache_bytes = max(self._stats.peak_cache_bytes, self._used_bytes)
        # Freshly-built entries are about to be activated; do NOT add to
        # LRU yet. They'll be added on first deactivate.

    def _evict_until_room(self, incoming_key: str, required: int) -> None:
        """Evict inactive LRU entries until ``required`` bytes fit. If
        active entries block sufficient eviction, raise
        :class:`ModelTooLargeError`."""
        if required > self._max_cache_bytes:
            raise ModelTooLargeError(
                key=incoming_key,
                required=required,
                used=self._used_bytes,
                limit=self._max_cache_bytes,
                active_refcounts={
                    k: e.active_count for k, e in self._entries.items() if e.active_count > 0
                },
            )
        while self._used_bytes + required > self._max_cache_bytes:
            try:
                victim = next(iter(self._lru))
            except StopIteration:
                raise ModelTooLargeError(
                    key=incoming_key,
                    required=required,
                    used=self._used_bytes,
                    limit=self._max_cache_bytes,
                    active_refcounts={
                        k: e.active_count for k, e in self._entries.items() if e.active_count > 0
                    },
                ) from None
            self._evict_inactive(victim)

    def _evict_inactive(self, key: str) -> None:
        entry = self._entries[key]
        assert entry.strategy is not None
        assert entry.active_count == 0
        bytes_freed = entry.cache_bytes
        # Detach from cache state — dropping `entry.strategy` releases
        # the cache's only reference to the strategy. If the entry's
        # factory built a fresh model (the typical pattern), the
        # strategy was the sole owner of the model and Python's
        # refcount-based GC frees the pinned tensors immediately.
        entry.strategy = None
        entry.cache_bytes = 0
        self._lru.pop(key, None)
        self._used_bytes -= bytes_freed
        self._stats.evictions += 1
        self._stats.bytes_evicted += bytes_freed
        self._after_release()

    def _evict_for_replace(self, key: str) -> None:
        """``register(replace=True)`` path. Refuses to clobber an active
        entry."""
        entry = self._entries[key]
        if entry.active_count > 0:
            raise ModelInUseError(
                f"cannot replace active registration {key!r} (refcount={entry.active_count})"
            )
        if entry.strategy is not None:
            self._evict_inactive(key)

    def _discard_entry(self, entry: _Entry, *, was_active: bool = False) -> None:
        """Detach an entry from cache state — for activation failures,
        deactivate poisoning, and unregistration. Dropping the
        ``entry.strategy`` reference triggers GC; if the cache was the
        sole owner of the model, pinned tensors are freed immediately.
        ``was_active=True`` also clears the active-module bookkeeping
        (only relevant on the deactivate-poisoned path; active state
        is otherwise managed by the use() context)."""
        key = entry.spec.key
        bytes_to_free = entry.cache_bytes
        entry.strategy = None
        entry.cache_bytes = 0
        if was_active:
            entry.active_value = None
            entry.active_count = 0
        self._lru.pop(key, None)
        self._used_bytes -= bytes_to_free
        self._after_release()

    def _after_release(self) -> None:
        if self._empty_host_cache is None:
            return
        try:
            self._empty_host_cache()
        except Exception:
            logger.warning("empty_host_cache callback raised; ignoring", exc_info=True)
