"""Policy-driven resource cache over :class:`CachedResource` plug-ins.

Manages cached pinned-CPU (or other resource-owned) backing storage for
multiple independent resources (models, LoRA adapters, etc.). When a new
resource needs more cache than is free, inactive entries are evicted
according to the configured :class:`EvictionPolicy` until room is
available. Active entries (currently inside a ``use()`` context) are
never evicted.

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
  discard the failed entry and propagate :class:`ActivationError`.
  Eviction is just a reference drop (no failure path) — pinned memory
  is freed when GC runs on the dropped strategy. **Factory failure**
  preserves the registration but leaves any pre-eviction *committed*
  — the cache evicts inactive entries to give the factory a
  predictable host-memory budget for pinning, and those evictions
  are not rolled back if the factory raises (rolling back would
  mean re-pinning the just-released weights, which can OOM the host
  allocator). The cache stays internally consistent; the cost is
  some warm cached entries disappearing.
- **No GPU budget.** The cache only enforces ``max_cache_bytes``
  (typically pinned host memory). It has a deliberately simple GPU
  placement guard: when callers pass a CUDA ``device``, at most one
  active key may occupy that CUDA device at a time.
- **Policy interfaces.** Host-cache eviction and active-lease placement
  are delegated to :class:`EvictionPolicy` and :class:`PlacementPolicy`
  instances; defaults preserve LRU eviction and one-key-per-CUDA-device
  placement.
- **Coarse thread-safety.** Public cache operations are protected by an
  instance lock. Factory/build, activation, and deactivation are
  serialized; the lock is released while caller code runs inside the
  ``use()`` context.

Instance-owned (not global) so it's library-friendly and embeddable.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast, overload

import torch

from ._devices import canonical_device
from .protocols import CachedResource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses and policy interfaces
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

    ``label`` is an optional human-readable name surfaced in errors or
    caller-managed logging; defaults to ``None`` if omitted.

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
    """Per-key state returned by :meth:`ModelCache.info`."""

    key: str
    label: str | None
    estimated_cache_bytes: int
    cache_bytes: int | None  # None when not built (registered but never used)
    cached: bool
    active_count: int


@dataclass(frozen=True, slots=True)
class EvictionCandidate:
    """Cache-owned view of one currently evictable inactive entry."""

    key: str
    cache_bytes: int
    estimated_cache_bytes: int
    label: str | None = None


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


@dataclass(frozen=True, slots=True)
class PlacementLease:
    """Placement reservation returned by a :class:`PlacementPolicy`.

    ``device`` is the concrete device passed to the resource's
    ``activate(device)`` call. It may be ``None`` for device-less
    resources. ``token`` is policy-private state returned unchanged to
    :meth:`PlacementPolicy.release`.
    """

    key: str
    device: torch.device | None
    token: object | None = None


class EvictionPolicy(Protocol):
    """Policy interface for inactive cached-entry eviction.

    The cache owns the authoritative candidate set and byte accounting.
    Policy state is advisory preference/history such as LRU order, LFU
    counters, priorities, or scoring weights.

    Methods are called by :class:`ModelCache` while holding its instance
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


class PlacementPolicy(Protocol):
    """Policy interface for active-lease placement.

    ``reserve`` may reject placement by raising :class:`ModelCacheError`
    or a subclass. The returned lease's ``device`` is used for
    activation and stored as the active device for re-entrant checks.
    ``requested_device`` is the raw caller request so custom policies
    may support sentinels such as ``"cuda:auto"``.
    """

    def reserve(self, *, key: str, requested_device: torch.device | str | None) -> PlacementLease:
        """Reserve placement for ``key`` and return the activation lease."""
        ...

    def validate_reentrant(
        self,
        *,
        key: str,
        active_device: torch.device | None,
        requested_device: torch.device | str | None,
    ) -> None:
        """Validate a nested same-key acquire on an already-active entry."""
        ...

    def release(self, lease: PlacementLease) -> None:
        """Release a lease returned by :meth:`reserve`."""
        ...


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


class ModelInUseError(ModelCacheError):
    """``evict(key)`` or ``clear()`` called while one or more entries
    are active."""


class EvictionPolicyError(ModelCacheError):
    """An :class:`EvictionPolicy` returned invalid or insufficient victims."""


class GpuDeviceOccupiedError(ModelCacheError):
    """A CUDA device already has an active model under the cache's
    one-active-key-per-GPU placement policy."""

    def __init__(self, *, device: torch.device, key: str, active_key: str) -> None:
        super().__init__(
            f"cannot activate {key!r} on {device}: device is already occupied by active model {active_key!r}"
        )
        self.device = device
        self.key = key
        self.active_key = active_key


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
        candidate_bytes = {candidate.key: candidate.cache_bytes for candidate in context.candidates}
        ordered_keys = [key for key in self._inactive_order if key in candidate_bytes]
        ordered_key_set = set(ordered_keys)
        ordered_keys.extend(candidate.key for candidate in context.candidates if candidate.key not in ordered_key_set)
        for key in ordered_keys:
            chosen.append(key)
            freed += candidate_bytes[key]
            if freed >= context.bytes_to_free:
                break
        return tuple(chosen)


class OneModelPerCudaDevicePolicy:
    """Default placement policy: one active model key per CUDA device."""

    def __init__(self) -> None:
        self._active_by_device: dict[torch.device, str] = {}

    def reserve(self, *, key: str, requested_device: torch.device | str | None) -> PlacementLease:
        device = canonical_device(requested_device) if requested_device is not None else None
        if device is None or device.type != "cuda":
            return PlacementLease(key=key, device=device)
        active_key = self._active_by_device.get(device)
        if active_key is not None and active_key != key:
            raise GpuDeviceOccupiedError(device=device, key=key, active_key=active_key)
        self._active_by_device[device] = key
        return PlacementLease(
            key=key,
            device=device,
            token=device,
        )

    def validate_reentrant(
        self,
        *,
        key: str,
        active_device: torch.device | None,
        requested_device: torch.device | str | None,
    ) -> None:
        device = canonical_device(requested_device) if requested_device is not None else None
        if device is not None and active_device is None:
            raise ModelCacheError(
                f"cannot acquire active {key!r} with device {device}: "
                "the existing lease was activated without a cache-visible device"
            )
        if device is not None and active_device is not None and device != active_device:
            raise ModelCacheError(f"cannot acquire active {key!r} on {device}: already active on {active_device}")

    def release(self, lease: PlacementLease) -> None:
        if lease.token is None:
            return
        device = cast(torch.device, lease.token)
        active_key = self._active_by_device.get(device)
        if active_key is None:
            return
        if active_key != lease.key:
            logger.warning(
                "GPU slot bookkeeping mismatch for %s: expected %r, found %r; leaving slot unchanged",
                device,
                lease.key,
                active_key,
            )
            return
        del self._active_by_device[device]


# ---------------------------------------------------------------------------
# Internal entry state
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    spec: ResourceSpec
    strategy: CachedResource | None = None
    cache_bytes: int = 0  # actual, post-build
    active_count: int = 0
    # Acquire-time placement for the current active lease. ``None`` means
    # the cache did not select a device for this resource.
    active_device: torch.device | None = None
    placement: PlacementLease | None = None
    # True while `pre_activate` is running. Guards against same-key
    # re-entry from inside the hook, which would otherwise activate
    # the strategy while we're still in the deactivated-config phase
    # and corrupt eviction-policy inactive state (active entry ending
    # up as an eviction candidate).
    configuring: bool = False


# ---------------------------------------------------------------------------
# ModelCache
# ---------------------------------------------------------------------------


class ModelCache:
    """Policy-driven pool over :class:`CachedResource` instances.

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
        post-activate contract violations, and deactivate failures.
        Flushes PyTorch's ``CachingHostAllocator`` so freed pinned
        pages return to the OS. If ``None`` and CUDA is available,
        defaults to ``torch._C._host_emptyCache`` when present. Pass
        a no-op callable to disable.
    eviction_policy:
        Optional inactive-entry eviction policy. Defaults to
        :class:`LRUEvictionPolicy`. The cache owns the policy instance
        and calls it while holding the cache lock.
    placement_policy:
        Optional active-lease placement policy. Defaults to
        :class:`OneModelPerCudaDevicePolicy`. The cache owns the policy
        instance and calls it while holding the cache lock.
    """

    def __init__(
        self,
        max_cache_bytes: int,
        *,
        empty_host_cache: Callable[[], None] | None = None,
        eviction_policy: EvictionPolicy | None = None,
        placement_policy: PlacementPolicy | None = None,
    ) -> None:
        if max_cache_bytes < 0:
            raise ValueError(f"max_cache_bytes must be >= 0, got {max_cache_bytes}")
        self._max_cache_bytes = max_cache_bytes
        self._empty_host_cache = self._resolve_host_cache_cb(empty_host_cache)
        self._entries: dict[str, _Entry] = {}
        self._eviction = eviction_policy if eviction_policy is not None else LRUEvictionPolicy()
        self._lock = threading.RLock()
        self._placement = placement_policy if placement_policy is not None else OneModelPerCudaDevicePolicy()
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
        """Current signed cache-budget headroom.

        Negative means a cached strategy grew after admission and the
        cache is currently over budget.
        """
        with self._lock:
            return self._available_cache_bytes

    @property
    def _available_cache_bytes(self) -> int:
        return self._max_cache_bytes - self._used_bytes

    def register(self, spec: ResourceSpec, *, replace: bool = False) -> None:
        """Register a lazy model factory without building it.

        Idempotent only with ``replace=True``: re-registering an
        existing key without ``replace`` raises
        :class:`DuplicateModelKeyError`. With ``replace=True``, any
        existing cached entry is evicted first (which raises
        :class:`ModelInUseError` if the key is active).
        """
        with self._lock:
            if spec.estimated_cache_bytes < 0:
                raise ValueError(f"spec.estimated_cache_bytes must be >= 0, got {spec.estimated_cache_bytes}")
            existing = self._entries.get(spec.key)
            if existing is not None:
                if not replace:
                    raise DuplicateModelKeyError(f"{spec.key!r} is already registered")
                if existing.active_count > 0:
                    raise ModelInUseError(
                        f"cannot replace active registration {spec.key!r} (refcount={existing.active_count})"
                    )
                if existing.strategy is not None:
                    self._evict_inactive(spec.key)
            self._entries[spec.key] = _Entry(spec=spec)

    def unregister(self, key: str, *, evict: bool = True) -> None:
        """Drop a registration. If a built strategy exists and
        ``evict=True`` (default), evict it; otherwise raise
        :class:`ModelInUseError` if it's cached or active."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            if entry.active_count > 0:
                raise ModelInUseError(f"cannot unregister active model {key!r} (refcount={entry.active_count})")
            if entry.strategy is not None:
                if not evict:
                    raise ModelInUseError(f"{key!r} has a built strategy; pass evict=True to release it")
                self._evict_inactive(key)
            del self._entries[key]

    @overload
    def use(
        self,
        model: ResourceSpec[T],
        *,
        device: torch.device | str | None = None,
        pre_activate: Callable[[CachedResource[T]], None] | None = None,
    ) -> contextlib.AbstractContextManager[T]: ...
    @overload
    def use(
        self,
        model: str,
        *,
        device: torch.device | str | None = None,
        pre_activate: Callable[[CachedResource[Any]], None] | None = None,
    ) -> contextlib.AbstractContextManager[Any]: ...

    @contextlib.contextmanager
    def use(
        self,
        model: str | ResourceSpec,
        *,
        device: torch.device | str | None = None,
        pre_activate: Callable[[CachedResource[Any]], None] | None = None,
    ) -> Iterator[Any]:
        """Acquire an active lease on a cached resource.

        Accepts either a registered key (string) or a :class:`ResourceSpec`
        (auto-registers if its key isn't already known). Yields the
        resource's :attr:`~CachedResource.value` for use; on context
        exit, releases the lease.

        ``device`` optionally selects placement for this acquire. It is
        passed to the strategy's :meth:`~CachedResource.activate` on the
        first lease. Re-entrant same-key acquires may omit ``device`` or
        must pass the same device; simultaneous activation of one cached
        entry on multiple devices is rejected. CUDA devices also
        participate in a simple placement guard: at most one active key
        may occupy a given CUDA device at a time.

        Re-entrant for the same key: nested ``use()`` calls bump a
        per-key refcount and share the already-active value (the
        underlying resource is *not* activated again).

        ``pre_activate`` is an optional hook called with the strategy
        on the first lease — after build (or cache hit) and *while the
        strategy is still deactivated*, before :meth:`activate`. Use
        for per-acquire configuration that requires the deactivated
        state (e.g., :meth:`ModelOffloader.set_loras`). The hook does
        NOT run on re-entrant nested acquires (the strategy is already
        active). A raising hook leaves the strategy state unknown: the
        entry is discarded and the exception is wrapped in
        :class:`ActivationError`.
        """
        spec = model if isinstance(model, ResourceSpec) else None
        key = spec.key if spec is not None else model
        assert isinstance(key, str)

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                if spec is None:
                    raise ModelNotRegisteredError(
                        f"{key!r} is not registered; pass a ResourceSpec to use() or call register() first"
                    )
                self.register(spec)
                entry = self._entries[key]

            value = self._acquire(
                entry,
                device=device,
                pre_activate=pre_activate,
            )
        try:
            yield value
        finally:
            with self._lock:
                self._release(entry)

    def evict(self, key: str) -> None:
        """Manually evict one inactive cached entry. Raises
        :class:`ModelInUseError` if active, no-op if not cached."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.strategy is None:
                return
            if entry.active_count > 0:
                raise ModelInUseError(f"cannot evict active model {key!r}")
            self._evict_inactive(key)

    def clear(self) -> None:
        """Evict all inactive entries. Registrations are preserved.
        Raises :class:`ModelInUseError` if any entry is active."""
        with self._lock:
            active = [k for k, e in self._entries.items() if e.active_count > 0]
            if active:
                raise ModelInUseError(f"cannot clear while models are active: {active}")
            for key in (candidate.key for candidate in self._eviction_candidates()):
                self._evict_inactive(key)

    def info(self, key: str) -> ModelInfo:
        """Return per-key state. Raises :class:`ModelNotRegisteredError`
        if unknown."""
        with self._lock:
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

    # ------------------------------------------------------------------
    # Lease lifecycle
    # ------------------------------------------------------------------

    def _acquire(
        self,
        entry: _Entry,
        *,
        device: torch.device | str | None = None,
        pre_activate: Callable[[CachedResource[Any]], None] | None = None,
    ) -> object:
        """Build (if needed), run ``pre_activate``, activate, and
        return the resource value.

        Re-entrant: if the entry is already active, bump the refcount
        and share the value without re-activating or re-running
        ``pre_activate``. Same-key re-entry from inside ``pre_activate``
        itself is rejected with :class:`ModelCacheError` to preserve
        eviction-policy inactive-state invariants. On pre_activate /
        activate / post-activate-reconcile failure, the entry is discarded
        and an exception
        propagates (pre_activate / activate failures are wrapped in
        :class:`ActivationError`)."""
        key = entry.spec.key

        # Same-key re-entry from inside the hook would acquire while
        # active_count is still 0, run a full activate/deactivate cycle
        # on the inner lease, mark the key inactive, then continue the
        # outer activate path — yielding an active entry that's also
        # tracked as evictable by the policy. The hook already has the
        # strategy as its argument; legitimate re-entry is for *different*
        # keys.
        if entry.configuring:
            raise ModelCacheError(
                f"cannot acquire {key!r} from inside its own pre_activate "
                "hook; the strategy is already available as the hook argument"
            )

        if entry.active_count > 0:
            return self._acquire_reentrant(entry, device)

        placement = self._placement.reserve(key=key, requested_device=device)
        try:
            strategy = self._prepare_strategy_for_activation(entry, pre_activate)
            value = self._activate_strategy(entry, strategy, placement.device)
        except BaseException:
            self._placement.release(placement)
            raise

        entry.active_count = 1
        entry.active_device = placement.device
        entry.placement = placement
        return value

    def _acquire_reentrant(self, entry: _Entry, device: torch.device | str | None) -> object:
        """Bump a same-key active lease without reactivating the strategy."""
        key = entry.spec.key
        self._placement.validate_reentrant(
            key=key,
            active_device=entry.active_device,
            requested_device=device,
        )
        strategy = entry.strategy
        assert strategy is not None
        value = strategy.value
        entry.active_count += 1
        return value

    def _prepare_strategy_for_activation(
        self,
        entry: _Entry,
        pre_activate: Callable[[CachedResource[Any]], None] | None,
    ) -> CachedResource[Any]:
        """Build or mark-active an inactive strategy, then run pre-activation config."""
        key = entry.spec.key
        if entry.strategy is None:
            self._build_into_entry(entry)
        else:
            self._eviction.mark_active(key)

        strategy = entry.strategy
        assert strategy is not None
        if pre_activate is not None:
            self._run_pre_activate(entry, strategy, pre_activate)
        return strategy

    def _run_pre_activate(
        self,
        entry: _Entry,
        strategy: CachedResource[Any],
        pre_activate: Callable[[CachedResource[Any]], None],
    ) -> None:
        entry.configuring = True
        try:
            pre_activate(strategy)
        except BaseException as exc:
            entry.configuring = False
            self._release_strategy(entry)
            raise ActivationError(f"pre_activate() failed for {entry.spec.key!r}") from exc
        entry.configuring = False

    def _activate_strategy(
        self,
        entry: _Entry,
        strategy: CachedResource[Any],
        device: torch.device | None,
    ) -> object:
        """Activate an inactive strategy and reconcile its budget accounting."""
        value = strategy.value
        try:
            if device is None:
                strategy.activate()
            else:
                strategy.activate(device)
        except BaseException as exc:
            entry.configuring = False
            self._release_strategy(entry)
            raise ActivationError(f"activate() failed for {entry.spec.key!r}") from exc

        try:
            self._reconcile_bytes(entry, strategy.cache_bytes, phase="activate")
        except BaseException:
            with contextlib.suppress(BaseException):
                strategy.deactivate()
            self._release_strategy(entry)
            raise
        return value

    def _release(self, entry: _Entry) -> None:
        """End a lease. On the final release, deactivate and mark the
        entry inactive for eviction policy state. A raising deactivate
        leaves the entry unrecoverable: discard and propagate so the
        caller sees the strategy's failure."""
        entry.active_count -= 1
        if entry.active_count > 0:
            return
        strategy = entry.strategy
        assert strategy is not None
        placement = entry.placement
        assert placement is not None
        try:
            strategy.deactivate()
        except BaseException:
            self._release_strategy(entry)
            self._placement.release(placement)
            raise
        entry.active_device = None
        entry.placement = None
        self._placement.release(placement)
        self._eviction.mark_inactive(entry.spec.key)

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
        estimate = entry.spec.estimated_cache_bytes
        self._evict_to_fit(estimate)
        strategy: CachedResource[Any] | None = entry.spec.factory()

        try:
            actual = strategy.cache_bytes
            if actual < 0:
                raise ModelCacheError(
                    f"strategy.cache_bytes for {entry.spec.key!r} returned {actual} (must be >= 0)",
                )
            if actual > estimate:
                self._evict_to_fit(actual)
        except BaseException:
            strategy = None
            self._after_release()
            raise

        assert strategy is not None
        entry.strategy = strategy
        # Initial commit: cache_bytes is currently 0; reconcile against
        # actual to record the bytes and update peak. Build-path
        # eviction already made room, so the over-budget warning path
        # is unreachable from here.
        self._reconcile_bytes(entry, actual, phase="build")
        # Freshly-built entries are about to be activated; do NOT mark
        # them inactive yet. That happens on first deactivate.

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
                f"strategy.cache_bytes for {entry.spec.key!r} returned {observed} (must be >= 0) after {phase}"
            )
        if observed == entry.cache_bytes:
            return
        delta = observed - entry.cache_bytes
        entry.cache_bytes = observed
        self._used_bytes += delta
        if self._used_bytes > self._max_cache_bytes:
            logger.warning(
                "ModelCache over budget after %s: %r grew to %d bytes; total %d/%d. "
                "The strategy reported a smaller cache_bytes earlier than at this "
                "point — usually a custom strategy that defers pinning. Pin in "
                "__init__ so cache_bytes is final at admission.",
                phase,
                entry.spec.key,
                observed,
                self._used_bytes,
                self._max_cache_bytes,
            )

    # ------------------------------------------------------------------
    # Eviction & release
    # ------------------------------------------------------------------

    def _eviction_candidates(self) -> tuple[EvictionCandidate, ...]:
        candidates: list[EvictionCandidate] = []
        for key, entry in self._entries.items():
            if entry.strategy is None or entry.active_count > 0:
                continue
            candidates.append(
                EvictionCandidate(
                    key=key,
                    cache_bytes=entry.cache_bytes,
                    estimated_cache_bytes=entry.spec.estimated_cache_bytes,
                    label=entry.spec.label,
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
        Raises :class:`ModelTooLargeError` when active entries block
        sufficient eviction."""
        if required_cache_bytes <= self._available_cache_bytes:
            return

        context = self._build_eviction_context(required_cache_bytes)
        bytes_to_free = context.bytes_to_free
        candidate_bytes = {candidate.key: candidate.cache_bytes for candidate in context.candidates}
        if sum(candidate_bytes.values()) < bytes_to_free:
            raise ModelTooLargeError(
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
        assert entry.strategy is not None
        assert entry.active_count == 0
        self._release_strategy(entry)

    def _release_strategy(self, entry: _Entry) -> None:
        """Drop the strategy reference and update accounting.

        Used on three paths: policy eviction, activation or contract
        failure, and deactivate failure. Dropping ``entry.strategy``
        triggers refcount-GC of pinned tensors when the cache was the
        sole owner; the host-cache flush runs after so freed pages
        return to the OS.
        """
        bytes_freed = entry.cache_bytes
        # Drop strategy ref BEFORE the host-cache flush so refcount-GC
        # frees pinned tensors in time for empty_host_cache to reclaim
        # them.
        entry.strategy = None
        entry.cache_bytes = 0
        entry.active_device = None
        entry.placement = None
        self._eviction.discard(entry.spec.key)
        self._used_bytes -= bytes_freed
        self._after_release()

    def _after_release(self) -> None:
        if self._empty_host_cache is None:
            return
        try:
            self._empty_host_cache()
        except Exception:
            logger.warning("empty_host_cache callback raised; ignoring", exc_info=True)
