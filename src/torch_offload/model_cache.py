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
from typing import Any, Generic, TypeVar, cast, overload

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
    """A strategy's ``activate()`` (or the ``pre_activate`` hook
    passed to :meth:`ModelCache.use`) raised. The cache discards the
    entry (drops the strategy reference, removes it from cache state)
    regardless of whether the entry was freshly built or previously
    cached — strategies with multi-step ``activate()`` (e.g.
    :func:`ModelOffloader`) can fail mid-way after partially
    installing hooks/pool/composed PinnedWeights, and caching such an
    entry as "ready to retry" lies about its state. The next acquire
    rebuilds via the registered factory. The original exception is
    chained via ``__cause__``."""


# ---------------------------------------------------------------------------
# Internal entry state
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    spec: ResourceSpec
    strategy: CachedResource | None = None
    cache_bytes: int = 0  # actual, post-build
    active_count: int = 0
    # True while `pre_activate` is running. Guards against same-key
    # re-entry from inside the hook, which would otherwise activate
    # the strategy while we're still in the deactivated-config phase
    # and corrupt the LRU invariant (active entry ending up in LRU).
    configuring: bool = False


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
        Optional callback invoked after any path that releases a
        cache-held strategy reference — eviction, admission rejection
        (negative or oversized post-build ``cache_bytes``), activation
        failure on freshly-built or previously-cached entries,
        post-activate contract violations, and deactivate poisoning.
        Flushes PyTorch's ``CachingHostAllocator`` so freed pinned
        pages return to the OS. If ``None`` and CUDA is available,
        defaults to ``torch._C._host_emptyCache`` when present. Pass
        a no-op callable to disable.
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
        host_empty: object = getattr(torch._C, "_host_emptyCache", None)
        if callable(host_empty) and torch.cuda.is_available():
            return cast(Callable[[], None], host_empty)
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
        existing = self._entries.get(spec.key)
        if existing is not None:
            if not replace:
                raise DuplicateModelKeyError(f"{spec.key!r} is already registered")
            if existing.active_count > 0:
                raise ModelInUseError(
                    f"cannot replace active registration {spec.key!r} "
                    f"(refcount={existing.active_count})"
                )
            if existing.strategy is not None:
                self._evict_inactive(spec.key)
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
    def use(
        self,
        model: ResourceSpec[T],
        *,
        pre_activate: Callable[[CachedResource[T]], None] | None = None,
    ) -> contextlib.AbstractContextManager[T]: ...
    @overload
    def use(
        self,
        model: str,
        *,
        pre_activate: Callable[[CachedResource[Any]], None] | None = None,
    ) -> contextlib.AbstractContextManager[Any]: ...

    @contextlib.contextmanager
    def use(
        self,
        model: str | ResourceSpec,
        *,
        pre_activate: Callable[[CachedResource[Any]], None] | None = None,
    ) -> Iterator[Any]:
        """Acquire an active lease on a cached resource.

        Accepts either a registered key (string) or a :class:`ResourceSpec`
        (auto-registers if its key isn't already known). Yields the
        resource's :attr:`~CachedResource.value` for use; on context
        exit, releases the lease.

        Re-entrant for the same key: nested ``use()`` calls bump a
        per-key refcount and share the already-active value (the
        underlying resource is *not* activated again).

        ``pre_activate`` is an optional hook called with the strategy
        on the first lease — after build (or LRU hit) and *while the
        strategy is still deactivated*, before :meth:`activate`. Use
        for per-acquire configuration that requires the deactivated
        state (e.g., :meth:`ModelOffloader.set_loras`). The hook does
        NOT run on re-entrant nested acquires (the strategy is already
        active). A raising hook is treated as activation poison: the
        entry is discarded and the exception is wrapped in
        :class:`ActivationError`.
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

        value = self._acquire(entry, pre_activate=pre_activate)
        try:
            yield value
        finally:
            self._release(entry)

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
    # Lease lifecycle
    # ------------------------------------------------------------------

    def _acquire(
        self,
        entry: _Entry,
        *,
        pre_activate: Callable[[CachedResource[Any]], None] | None = None,
    ) -> object:
        """Build (if needed), run ``pre_activate``, activate, and
        return the resource value.

        Re-entrant: if the entry is already active, bump the refcount
        and share the value without re-activating or re-running
        ``pre_activate``. Same-key re-entry from inside ``pre_activate``
        itself is rejected with :class:`ModelCacheError` to preserve
        the LRU invariant. On pre_activate / activate / post-activate-
        reconcile failure, the entry is discarded and an exception
        propagates (pre_activate / activate failures are wrapped in
        :class:`ActivationError`)."""
        key = entry.spec.key

        # Same-key re-entry from inside the hook would acquire while
        # active_count is still 0, run a full activate/deactivate cycle
        # on the inner lease, leave the key in LRU, then continue the
        # outer activate path — yielding an active entry that's also
        # in LRU. A later eviction would assert. The hook already has
        # the strategy as its argument; legitimate re-entry is for
        # *different* keys.
        if entry.configuring:
            raise ModelCacheError(
                f"cannot acquire {key!r} from inside its own pre_activate "
                "hook; the strategy is already available as the hook argument"
            )

        # Re-entrant lease: strategy is already active. The protocol
        # contract guarantees `value` is stable across activation
        # cycles, so re-reading it is safe. pre_activate is skipped —
        # the strategy is active, can't be reconfigured, and the
        # caller's hook would either fail or silently violate
        # invariants.
        if entry.active_count > 0:
            strategy = entry.strategy
            assert strategy is not None
            value = strategy.value
            entry.active_count += 1
            return value

        if entry.strategy is None:
            self._build_into_entry(entry)
        else:
            self._stats.hits += 1
            self._lru.pop(key, None)  # leaving the inactive set

        strategy = entry.strategy
        assert strategy is not None

        # pre_activate runs while the strategy is still deactivated —
        # this is the user's hook for per-acquire configuration that
        # requires the deactivated state (e.g., set_loras). A raising
        # hook leaves the strategy in an unknown state; treat as
        # activation poison. The `configuring` flag locks out same-key
        # re-entry through user code calling back into `cache.use()`.
        if pre_activate is not None:
            entry.configuring = True
            try:
                pre_activate(strategy)
            except BaseException as exc:
                entry.configuring = False
                self._stats.activation_errors += 1
                self._release_strategy(entry, evicted=False)
                raise ActivationError(
                    f"pre_activate() failed for {key!r}"
                ) from exc
            entry.configuring = False

        # Read value BEFORE activate. The protocol says `value` is
        # available regardless of activation state; reading first
        # eliminates a post-activate exception window where a raising
        # `value` getter would skip the deactivate path on the now-
        # active strategy.
        value = strategy.value
        try:
            strategy.activate()
        except BaseException as exc:
            self._stats.activation_errors += 1
            self._release_strategy(entry, evicted=False)
            raise ActivationError(f"activate() failed for {key!r}") from exc

        try:
            self._reconcile_bytes(entry, strategy.cache_bytes, phase="activate")
        except BaseException:
            with contextlib.suppress(BaseException):
                strategy.deactivate()
            self._stats.activation_errors += 1
            self._release_strategy(entry, evicted=False)
            raise

        entry.active_count = 1
        return value

    def _release(self, entry: _Entry) -> None:
        """End a lease. On the final release, deactivate and re-enter
        the entry into the LRU at MRU. A raising deactivate poisons
        the entry — discard and propagate so the caller sees the
        strategy's failure."""
        entry.active_count -= 1
        if entry.active_count > 0:
            return
        strategy = entry.strategy
        assert strategy is not None
        try:
            strategy.deactivate()
        except BaseException:
            self._release_strategy(entry, evicted=False)
            raise
        self._lru[entry.spec.key] = None

    # ------------------------------------------------------------------
    # Build & accounting
    # ------------------------------------------------------------------

    def _build_into_entry(self, entry: _Entry) -> None:
        """Cache miss: pre-evict, build, validate, commit accounting.

        Pre-build eviction trusts the estimate. After construction,
        a negative ``cache_bytes`` is rejected and an over-estimate
        triggers further eviction (which can fail with
        :class:`ModelTooLargeError`). All post-factory failure paths
        drop the local strategy ref BEFORE the host-cache flush so
        refcount-GC frees pinned tensors in time for the flush to
        actually reclaim them.
        """
        key = entry.spec.key
        self._stats.misses += 1
        estimate = entry.spec.estimated_cache_bytes
        self._evict_until_room(key, estimate)
        try:
            strategy = entry.spec.factory()
        except BaseException:
            self._stats.factory_errors += 1
            raise

        try:
            actual = strategy.cache_bytes
            if actual < 0:
                raise ModelCacheError(
                    f"strategy.cache_bytes for {key!r} returned {actual} (must be >= 0)"
                )
            if actual > estimate:
                self._evict_until_room(key, actual)
        except BaseException:
            del strategy
            self._after_release()
            raise

        entry.strategy = strategy
        # Initial commit: cache_bytes is currently 0; reconcile against
        # actual to record the bytes and update peak. Build-path
        # eviction already made room, so the over-budget warning path
        # is unreachable from here.
        self._reconcile_bytes(entry, actual, phase="build")
        self._stats.builds += 1
        # Freshly-built entries are about to be activated; do NOT add
        # to LRU yet. They'll be added on first deactivate.

    def _reconcile_bytes(self, entry: _Entry, observed: int, *, phase: str) -> None:
        """Apply ``observed`` to entry and global byte accounting.

        Used on the build path (initial commit, 0 → actual) and on the
        activate path (drift from a strategy that defers pinning).
        Raises :class:`ModelCacheError` if ``observed < 0`` so a
        misbehaving strategy can't corrupt ``_used_bytes`` (the build
        path also pre-validates before attaching to keep refcount-GC
        timing correct on the cleanup path). Logs a warning if the
        update pushes total usage over budget — only reachable from
        the activate path; build-path pre-eviction makes the budget
        already satisfied at this call.
        """
        if observed < 0:
            raise ModelCacheError(
                f"strategy.cache_bytes for {entry.spec.key!r} returned "
                f"{observed} (must be >= 0) after {phase}"
            )
        if observed == entry.cache_bytes:
            return
        delta = observed - entry.cache_bytes
        entry.cache_bytes = observed
        self._used_bytes += delta
        self._stats.peak_cache_bytes = max(self._stats.peak_cache_bytes, self._used_bytes)
        if self._used_bytes > self._max_cache_bytes:
            logger.warning(
                "ModelCache over budget after %s: %r grew to %d bytes; total %d/%d. "
                "The strategy reported a smaller cache_bytes earlier than at this "
                "point — usually a custom strategy that defers pinning. Pin in "
                "__init__ so cache_bytes is final at admission.",
                phase, entry.spec.key, observed,
                self._used_bytes, self._max_cache_bytes,
            )

    # ------------------------------------------------------------------
    # Eviction & release
    # ------------------------------------------------------------------

    def _evict_until_room(self, incoming_key: str, required: int) -> None:
        """Evict inactive LRU entries until ``required`` bytes fit.
        Raises :class:`ModelTooLargeError` when active entries block
        sufficient eviction."""
        if required > self._max_cache_bytes:
            raise self._too_large(incoming_key, required)
        while self._used_bytes + required > self._max_cache_bytes:
            try:
                victim = next(iter(self._lru))
            except StopIteration:
                raise self._too_large(incoming_key, required) from None
            self._evict_inactive(victim)

    def _too_large(self, key: str, required: int) -> ModelTooLargeError:
        return ModelTooLargeError(
            key=key,
            required=required,
            used=self._used_bytes,
            limit=self._max_cache_bytes,
            active_refcounts={
                k: e.active_count
                for k, e in self._entries.items()
                if e.active_count > 0
            },
        )

    def _evict_inactive(self, key: str) -> None:
        """Release a cached, inactive entry as an eviction. Asserts the
        precondition (built + inactive); :meth:`_release_strategy`
        bumps eviction stats."""
        entry = self._entries[key]
        assert entry.strategy is not None
        assert entry.active_count == 0
        self._release_strategy(entry, evicted=True)

    def _release_strategy(self, entry: _Entry, *, evicted: bool) -> None:
        """Drop the strategy reference and update accounting.

        Used on three paths: LRU eviction (``evicted=True``), activation
        or contract failure (``evicted=False``), and deactivate poisoning
        (``evicted=False``). Dropping ``entry.strategy`` triggers
        refcount-GC of pinned tensors when the cache was the sole
        owner; the host-cache flush runs after so freed pages return
        to the OS.
        """
        bytes_freed = entry.cache_bytes
        # Drop strategy ref BEFORE the host-cache flush so refcount-GC
        # frees pinned tensors in time for empty_host_cache to reclaim
        # them.
        entry.strategy = None
        entry.cache_bytes = 0
        self._lru.pop(entry.spec.key, None)
        self._used_bytes -= bytes_freed
        if evicted:
            self._stats.evictions += 1
            self._stats.bytes_evicted += bytes_freed
        self._after_release()

    def _after_release(self) -> None:
        if self._empty_host_cache is None:
            return
        try:
            self._empty_host_cache()
        except Exception:
            logger.warning("empty_host_cache callback raised; ignoring", exc_info=True)
