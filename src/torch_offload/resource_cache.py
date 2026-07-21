"""Policy-driven cache for reusable resource stores.

Manages cached pinned-CPU (or other resource-owned) backing storage for
multiple independent resources (models, LoRA adapters, etc.). When a new
store needs more cache than is free, inactive entries are evicted
according to the configured :class:`EvictionPolicy` until room is
available. Leased entries are never evicted.

Design highlights
-----------------
- **Store-only.** The cache builds, budgets, leases, and evicts
  :class:`ResourceStore` instances. It does not create runtime bindings,
  choose devices, or know how one resource is applied to another.
- **Structural specs.** Cache entries implement :class:`ResourceSpec`; concrete
  specs are independent frozen dataclasses rather than an inheritance tree.
- **Reference-counted leases.** A lease protects its store from eviction.
  Multiple callers may lease the same cached store concurrently; whether its
  value supports overlapping use is a separate policy owned by that resource.
- **Transactional admission, with one caveat.** Eviction is just a
  reference drop (no failure path) — pinned memory is freed when GC runs
  on the dropped store. **Store factory failure**
  preserves the registration but leaves any pre-eviction *committed*
  — the cache evicts inactive entries to give the factory a
  predictable host-memory budget for pinning, and those evictions
  are not rolled back if the factory raises (rolling back would
  mean re-pinning the just-released weights, which can OOM the host
  allocator). The cache stays internally consistent; the cost is
  some warm cached entries disappearing.
- **No runtime or GPU lifecycle.** The cache only enforces
  ``max_cache_bytes`` (typically pinned host memory). Activation and GPU
  residency belong to resources and consumers such as
  :class:`~torch_offload.ModelOffloader` and
  :class:`~torch_offload.CachedModelRunner`.
- **Policy interface.** Host-cache eviction is delegated to an
  :class:`EvictionPolicy`; the default preserves LRU eviction.
- **Coarse thread-safety.** Public cache operations are protected by an
  instance lock. Factory/build and lease accounting are serialized; the
  lock is released while caller code holds a lease.

Instance-owned (not global) so it's library-friendly and embeddable.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast

import torch

from .protocols import ResourceSpec, ResourceStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public dataclasses and policy interfaces
# ---------------------------------------------------------------------------


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ResourceInfo:
    """Per-key state returned by :meth:`ResourceCache.info`."""

    key: str
    estimated_cache_bytes: int
    cache_bytes: int | None  # None when not built (registered but never used)
    cached: bool
    lease_count: int


@dataclass(frozen=True, slots=True)
class EvictionCandidate:
    """Cache-owned view of one currently evictable inactive entry."""

    key: str
    cache_bytes: int
    estimated_cache_bytes: int


@dataclass(frozen=True, slots=True)
class EvictionContext:
    """Read-only eviction decision context passed to an eviction policy."""

    required_cache_bytes: int
    used_cache_bytes: int
    max_cache_bytes: int
    candidates: tuple[EvictionCandidate, ...]

    @property
    def bytes_to_free(self) -> int:
        """Minimum cache bytes that must be evicted to admit the incoming entry."""
        return max(
            0,
            self.used_cache_bytes + self.required_cache_bytes - self.max_cache_bytes,
        )


class EvictionPolicy(Protocol):
    """Policy interface for inactive cached-entry eviction.

    The cache owns the authoritative candidate set and byte accounting.
    Policy state is advisory preference/history such as LRU order, LFU
    counters, priorities, or scoring weights.

    Methods are called by :class:`ResourceCache` while holding its instance
    lock, so policy implementations do not need their own locking unless
    they are shared outside the cache.
    """

    def mark_active(self, key: str) -> None:
        """Record that ``key`` is no longer inactive."""
        ...

    def mark_inactive(self, key: str) -> None:
        """Record that ``key`` became inactive."""
        ...

    def discard(self, key: str) -> None:
        """Remove ``key`` from all policy state."""
        ...

    def choose_victims(self, context: EvictionContext) -> tuple[str, ...]:
        """Return unique candidate keys totaling enough bytes to admit the entry."""
        ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CacheError(RuntimeError):
    """Base for all cache errors."""


class ResourceNotRegisteredError(CacheError):
    """``lease(str)`` called for a key that has no registration."""


class DuplicateResourceKeyError(CacheError):
    """``register(spec)`` called for a key that is already registered
    (without ``replace=True``)."""


class ResourceTooLargeError(CacheError):
    """Even after evicting all inactive entries, the requested resource
    cannot fit in the budget."""

    def __init__(
        self,
        *,
        required: int,
        used: int,
        limit: int,
    ) -> None:
        short = used + required - limit
        super().__init__(
            f"Cannot admit resource: needs {required} bytes, {used}/{limit} already used (short by {short})."
        )
        self.required = required
        self.used = used
        self.limit = limit


class ResourceLeasedError(CacheError):
    """A cache mutation targeted one or more currently leased entries."""


class ResourceCachedError(CacheError):
    """A registration mutation required an explicit store eviction."""


class EvictionPolicyError(CacheError):
    """An :class:`EvictionPolicy` returned invalid or insufficient victims."""


# ---------------------------------------------------------------------------
# Default policy implementations
# ---------------------------------------------------------------------------


class LRUEvictionPolicy:
    """Default eviction policy: evict least-recently-used inactive keys."""

    def __init__(self) -> None:
        self._inactive_order: OrderedDict[str, None] = OrderedDict()

    def mark_active(self, key: str) -> None:
        self._inactive_order.pop(key, None)

    def mark_inactive(self, key: str) -> None:
        self._inactive_order.pop(key, None)
        self._inactive_order[key] = None

    def discard(self, key: str) -> None:
        self._inactive_order.pop(key, None)

    def choose_victims(self, context: EvictionContext) -> tuple[str, ...]:
        if context.bytes_to_free <= 0:
            return ()
        chosen: list[str] = []
        freed = 0
        # Zero-byte entries (e.g. ordinary heap-object wrappers)
        # free nothing, so evicting them can never help admit an entry.
        candidate_bytes = {
            candidate.key: candidate.cache_bytes for candidate in context.candidates if candidate.cache_bytes > 0
        }
        ordered_keys = [key for key in self._inactive_order if key in candidate_bytes]
        ordered_key_set = set(ordered_keys)
        ordered_keys.extend(key for key in candidate_bytes if key not in ordered_key_set)
        for key in ordered_keys:
            chosen.append(key)
            freed += candidate_bytes[key]
            if freed >= context.bytes_to_free:
                break
        return tuple(chosen)


# ---------------------------------------------------------------------------
# Internal entry state
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    spec: ResourceSpec[Any]
    store: ResourceStore | None = None
    cache_bytes: int = 0  # actual, post-build
    lease_count: int = 0

    @property
    def in_use(self) -> bool:
        return self.lease_count > 0


# ---------------------------------------------------------------------------
# ResourceCache
# ---------------------------------------------------------------------------


class ResourceCache:
    """Policy-driven pool over :class:`ResourceStore` instances.

    Parameters
    ----------
    max_cache_bytes:
        Total budget for cached resource backing storage (typically
        pinned host memory). Must be ≥ 0.
    empty_host_cache:
        Optional callback invoked after any path that releases a
        cache-held store reference — eviction and admission rejection
        (negative or oversized post-build ``cache_bytes``).
        Flushes PyTorch's ``CachingHostAllocator`` so freed pinned
        pages return to the OS. If ``None`` and CUDA is available,
        defaults to ``torch._C._host_emptyCache`` when present. Pass
        a no-op callable to disable.
    eviction_policy:
        Optional inactive-entry eviction policy. Defaults to
        :class:`LRUEvictionPolicy`. The cache owns the policy instance
        and calls it while holding the cache lock.
    """

    def __init__(
        self,
        max_cache_bytes: int,
        *,
        empty_host_cache: Callable[[], None] | None = None,
        eviction_policy: EvictionPolicy | None = None,
    ) -> None:
        if max_cache_bytes < 0:
            raise ValueError(f"max_cache_bytes must be >= 0, got {max_cache_bytes}")
        self._max_cache_bytes = max_cache_bytes
        self._empty_host_cache = self._resolve_host_cache_cb(empty_host_cache)
        self._entries: dict[str, _Entry] = {}
        self._eviction = eviction_policy if eviction_policy is not None else LRUEvictionPolicy()
        self._lock = threading.RLock()
        self._used_bytes = 0

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
        with self._lock:
            return self._used_bytes

    @property
    def available_cache_bytes(self) -> int:
        """Current cache-budget headroom."""
        with self._lock:
            return self._available_cache_bytes

    @property
    def _available_cache_bytes(self) -> int:
        return self._max_cache_bytes - self._used_bytes

    def register(self, spec: ResourceSpec[Any], *, replace: bool = False) -> None:
        """Register a lazy store factory without building it.

        Idempotent only with ``replace=True``: re-registering an
        existing key without ``replace`` raises
        :class:`DuplicateResourceKeyError`. With ``replace=True``, any
        existing cached entry is evicted first (which raises
        :class:`ResourceLeasedError` if the key is leased).
        """
        with self._lock:
            if spec.estimated_cache_bytes < 0:
                raise ValueError(f"spec.estimated_cache_bytes must be >= 0, got {spec.estimated_cache_bytes}")
            existing = self._entries.get(spec.key)
            if existing is not None:
                if not replace:
                    raise DuplicateResourceKeyError(f"{spec.key!r} is already registered")
                if existing.in_use:
                    raise ResourceLeasedError(
                        f"cannot replace leased registration {spec.key!r} (leases={existing.lease_count})"
                    )
                if existing.store is not None:
                    self._evict_inactive(spec.key)
            self._entries[spec.key] = _Entry(spec=spec)

    def unregister(self, key: str, *, evict: bool = True) -> None:
        """Drop a registration. If a built store exists and
        ``evict=True`` (default), evict it; otherwise raise
        :class:`ResourceCachedError` if eviction is disabled, or
        :class:`ResourceLeasedError` if it is leased."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            if entry.in_use:
                raise ResourceLeasedError(
                    f"cannot unregister leased resource {key!r} (leases={entry.lease_count})"
                )
            if entry.store is not None:
                if not evict:
                    raise ResourceCachedError(
                        f"{key!r} has a built store; pass evict=True to release it"
                    )
                self._evict_inactive(key)
            del self._entries[key]

    @contextlib.contextmanager
    def lease(
        self,
        resource: str | ResourceSpec[T],
    ) -> Iterator[T]:
        """Lease a cached value, building its store on a cache miss.

        A lease protects the store from eviction for the duration of the
        context. Runtime binding and device activation are caller-owned.
        Passing a spec auto-registers it when its key is not yet known.
        """
        with self._lock:
            entry = self._get_entry(resource)
            self._ensure_store(entry)
            store = entry.store
            assert store is not None
            entry.lease_count += 1
            self._eviction.mark_active(entry.spec.key)
        try:
            yield entry.spec.value(store)
        finally:
            with self._lock:
                entry.lease_count -= 1
                self._mark_inactive_if_unleased(entry)

    @contextlib.contextmanager
    def lease_many(
        self,
        resources: Sequence[str | ResourceSpec[Any]],
    ) -> Iterator[tuple[Any, ...]]:
        """Lease resources in order and release them in reverse order.

        Earlier resources are protected before later stores are admitted,
        making dependent admission safe without teaching the cache about
        the dependency itself.
        """
        with contextlib.ExitStack() as stack:
            yield tuple(stack.enter_context(self.lease(resource)) for resource in resources)

    def evict(self, key: str) -> None:
        """Manually evict one unleased cached entry. Raises
        :class:`ResourceLeasedError` if leased, no-op if not cached."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.store is None:
                return
            if entry.in_use:
                raise ResourceLeasedError(f"cannot evict leased resource {key!r}")
            self._evict_inactive(key)

    def clear(self) -> None:
        """Evict all unleased entries. Registrations are preserved.
        Raises :class:`ResourceLeasedError` if any entry is leased."""
        with self._lock:
            active = [k for k, e in self._entries.items() if e.in_use]
            if active:
                raise ResourceLeasedError(
                    f"cannot clear while resources are leased: {active}"
                )
            for key in (candidate.key for candidate in self._eviction_candidates()):
                self._evict_inactive(key)

    def info(self, key: str) -> ResourceInfo:
        """Return per-key state. Raises :class:`ResourceNotRegisteredError`
        if unknown."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                raise ResourceNotRegisteredError(f"{key!r} is not registered")
            return ResourceInfo(
                key=entry.spec.key,
                estimated_cache_bytes=entry.spec.estimated_cache_bytes,
                cache_bytes=entry.cache_bytes if entry.store is not None else None,
                cached=entry.store is not None,
                lease_count=entry.lease_count,
            )

    # ------------------------------------------------------------------
    # Lease lifecycle
    # ------------------------------------------------------------------

    def _get_entry(self, resource: str | ResourceSpec[Any]) -> _Entry:
        """Resolve a resource key/spec to its cache entry, auto-registering
        specs that are not known yet."""
        spec = None if isinstance(resource, str) else resource
        key = resource if isinstance(resource, str) else resource.key

        entry = self._entries.get(key)
        if entry is not None:
            return entry
        if spec is None:
            raise ResourceNotRegisteredError(
                f"{key!r} is not registered; pass a ResourceSpec to lease() or call register() first"
            )
        self.register(spec)
        return self._entries[key]

    def _ensure_store(self, entry: _Entry) -> None:
        """Build and cache the store if this is a cache miss."""
        if entry.store is None:
            self._build_into_entry(entry)

    def _mark_inactive_if_unleased(self, entry: _Entry) -> None:
        if entry.store is not None and not entry.in_use:
            self._eviction.mark_inactive(entry.spec.key)

    # ------------------------------------------------------------------
    # Build & accounting
    # ------------------------------------------------------------------

    def _build_into_entry(self, entry: _Entry) -> None:
        """Cache miss: pre-evict, build, validate, commit accounting.

        Pre-build eviction trusts the estimate. After construction,
        a negative ``cache_bytes`` is rejected and an over-estimate
        triggers further eviction (which can fail with
        :class:`ResourceTooLargeError`). All post-factory failure paths
        drop the local store ref BEFORE the host-cache flush so
        refcount-GC frees pinned tensors in time for the flush to
        actually reclaim them.
        """
        estimate = entry.spec.estimated_cache_bytes
        self._evict_to_fit(estimate)
        store: ResourceStore | None = entry.spec.build_store()

        try:
            actual = store.cache_bytes
            if actual < 0:
                raise CacheError(
                    f"store.cache_bytes for {entry.spec.key!r} returned {actual} (must be >= 0)",
                )
            if actual > estimate:
                self._evict_to_fit(actual)
        except BaseException:
            store = None
            self._after_release()
            raise

        assert store is not None
        entry.store = store
        entry.cache_bytes = actual
        self._used_bytes += actual
        self._eviction.mark_inactive(entry.spec.key)

    # ------------------------------------------------------------------
    # Eviction & release
    # ------------------------------------------------------------------

    def _eviction_candidates(self) -> tuple[EvictionCandidate, ...]:
        candidates: list[EvictionCandidate] = []
        for key, entry in self._entries.items():
            if entry.store is None or entry.in_use:
                continue
            candidates.append(
                EvictionCandidate(
                    key=key,
                    cache_bytes=entry.cache_bytes,
                    estimated_cache_bytes=entry.spec.estimated_cache_bytes,
                )
            )
        return tuple(candidates)

    def _build_eviction_context(self, required_cache_bytes: int) -> EvictionContext:
        return EvictionContext(
            required_cache_bytes=required_cache_bytes,
            used_cache_bytes=self._used_bytes,
            max_cache_bytes=self._max_cache_bytes,
            candidates=self._eviction_candidates(),
        )

    def _evict_to_fit(self, required_cache_bytes: int) -> None:
        """Evict inactive entries so ``required_cache_bytes`` fits.
        Raises :class:`ResourceTooLargeError` when leased entries block
        sufficient eviction."""
        if required_cache_bytes <= self._available_cache_bytes:
            return

        context = self._build_eviction_context(required_cache_bytes)
        bytes_to_free = context.bytes_to_free
        candidate_bytes = {candidate.key: candidate.cache_bytes for candidate in context.candidates}
        if sum(candidate_bytes.values()) < bytes_to_free:
            raise ResourceTooLargeError(
                required=required_cache_bytes,
                used=self._used_bytes,
                limit=self._max_cache_bytes,
            )

        victims = self._eviction.choose_victims(context)
        if len(victims) != len(set(victims)) or not set(victims) <= candidate_bytes.keys():
            raise EvictionPolicyError("eviction policy chose invalid victims")

        selected_bytes = sum(candidate_bytes[victim] for victim in victims)
        if selected_bytes < bytes_to_free:
            raise EvictionPolicyError(
                f"eviction policy chose insufficient victims selected {selected_bytes} bytes, need {bytes_to_free}",
            )

        for victim in victims:
            if required_cache_bytes <= self._available_cache_bytes:
                break
            self._evict_inactive(victim)

    def _evict_inactive(self, key: str) -> None:
        """Release a cached, inactive entry as an eviction. Asserts the
        precondition (built + inactive)."""
        entry = self._entries[key]
        assert entry.store is not None
        assert not entry.in_use
        self._release_store(entry)

    def _release_store(self, entry: _Entry) -> None:
        """Drop the store reference and update accounting.

        Used for policy eviction and store admission rejection. Dropping
        ``entry.store`` triggers refcount-GC of pinned tensors when the
        cache was the sole owner; the host-cache flush runs after so
        freed pages return to the OS.
        """
        bytes_freed = entry.cache_bytes
        # Drop store ref BEFORE the host-cache flush so refcount-GC
        # frees pinned tensors in time for empty_host_cache to reclaim
        # them.
        entry.store = None
        entry.cache_bytes = 0
        self._eviction.discard(entry.spec.key)
        self._used_bytes -= bytes_freed
        if bytes_freed > 0:
            self._after_release()

    def _after_release(self) -> None:
        if self._empty_host_cache is None:
            return
        try:
            self._empty_host_cache()
        except Exception:
            logger.warning("empty_host_cache callback raised; ignoring", exc_info=True)
