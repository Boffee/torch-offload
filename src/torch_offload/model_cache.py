"""Policy-driven resource cache over store/binding plug-ins.

Manages cached pinned-CPU (or other resource-owned) backing storage for
multiple independent resources (models, LoRA adapters, etc.). When a new
store needs more cache than is free, inactive entries are evicted
according to the configured :class:`EvictionPolicy` until room is
available. Active entries (currently inside a ``use()`` context) are
never evicted.

Design highlights
-----------------
- **Resource-agnostic.** The cache only talks to
  :class:`ResourceStore` for cached backing bytes and
  :class:`ResourceBinding` for per-use activation. Pluggable: today
  :class:`ModelOffloader`, :class:`MpsWeights`, and :class:`LoRA`;
  future resources (disk-mmap, NVMe-paged, multi-GPU shard) just satisfy
  the protocol pair.
- **Active bindings.** Multiple keys can be active simultaneously
  (e.g. text encoder and embedding processor in the same call), and the
  same key can be acquired multiple times when the spec opts in via
  :attr:`ResourceSpec.allow_concurrent_binding`. The default
  :class:`ModelSpec` binds the cached store's canonical model and
  rejects same-key concurrent bindings; passing ``skeleton_factory=...``
  enables per-bind module instances and concurrent bindings for frozen
  stores. Trainable model stores always reject concurrent same-key
  bindings so optimizer parameter identity stays unambiguous.
- **Transactional admission, with one caveat.** Activation failures
  do not publish the failed binding and propagate the original exception.
  Eviction is just a reference drop (no failure path) — pinned memory
  is freed when GC runs on the dropped store. **Store factory failure**
  preserves the registration but leaves any pre-eviction *committed*
  — the cache evicts inactive entries to give the factory a
  predictable host-memory budget for pinning, and those evictions
  are not rolled back if the factory raises (rolling back would
  mean re-pinning the just-released weights, which can OOM the host
  allocator). The cache stays internally consistent; the cost is
  some warm cached entries disappearing.
- **No GPU budget.** The cache only enforces ``max_cache_bytes``
  (typically pinned host memory). GPU residency is caller-managed:
  multiple active keys may target the same device when the caller knows
  they fit.
- **Policy interface.** Host-cache eviction is delegated to an
  :class:`EvictionPolicy`; the default preserves LRU eviction.
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
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, cast

import torch
from torch import nn

from ._devices import canonical_device
from .lora import LoRA
from .model_offloader import LoraMode, ModelOffloader, ModelOffloaderStore
from .protocols import ResourceBinding, ResourceStore
from .stream_config import StreamConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses and policy interfaces
# ---------------------------------------------------------------------------


T = TypeVar("T")
M = TypeVar("M", bound=nn.Module)


def _allow_concurrent_default(_store: ResourceStore) -> bool:
    return True


@dataclass(frozen=True)
class ResourceSpec(Generic[T]):
    """Registration spec for a cached resource.

    ``key`` is the cache identity — caller-chosen and must include any
    construction inputs that affect the resulting weights or structure
    (checkpoint hash, dtype, quantization config, LoRA stack, etc.). The
    cache does not introspect the factory; calling ``use()`` with the
    same key but a different factory silently returns the cached entry.

    ``estimated_cache_bytes`` is used to evict before building. The
    cache reconciles against the actual ``store.cache_bytes`` after
    construction; if the actual exceeds the estimate enough to overflow
    the budget, the cache evicts more or rejects the admission.

    ``allow_concurrent_binding`` is called by the cache before issuing a
    second active binding for the same key. Defaults to always allow;
    specs that need single-binding semantics (e.g. trainable models, or
    frozen models without a per-bind skeleton factory) return ``False``.

    .. note::
       The ``store_factory`` should build a *fresh* store that the cache
       solely owns. Eviction releases backing storage only when the store
       becomes unreachable — Python's refcount-based GC handles this
       when the cache is the sole owner.
    """

    key: str
    estimated_cache_bytes: int
    store_factory: Callable[[], ResourceStore]
    bind: Callable[[ResourceStore], ResourceBinding[T]]
    allow_concurrent_binding: Callable[[ResourceStore], bool] = _allow_concurrent_default


class ModelSpec(ResourceSpec[M]):
    """Model-specific cache spec built from one user model factory.

    ``factory`` runs normally once to construct the cached
    :class:`ModelOffloaderStore`. Each ``use()`` then binds the cached
    pinned/streamed bytes into a module instance. The spec is generic
    over the factory's module type, so ``cache.use(spec)`` yields that
    concrete type rather than bare :class:`nn.Module`.

    By default, every ``use()`` reuses the canonical model instance
    owned by the cached store and the cache rejects a second concurrent
    binding for the same key. This matches the trainable-model contract
    (optimizer parameter identity stays stable) and removes the hidden
    requirement that ``factory`` respect ``torch.device("meta")``.

    Pass ``skeleton_factory`` to enable multiple concurrent bindings
    (e.g. the same model bound on two GPUs at once). The skeleton factory
    is called once per ``use()`` and must produce a module with the same
    parameter and buffer structure as ``factory``'s output — the cached
    bytes are re-bound into the fresh module. Placeholder dtypes need not
    match for plain tensors (binding overwrites them with store-backed
    storage), so a config-built skeleton binds against natively loaded
    weights; shapes and quantized-wrapper representations must match.
    Common forms:

    - ``skeleton_factory=lambda: build_under_meta(factory)`` for an
      allocation-light meta skeleton (only when ``factory`` respects
      ``torch.device("meta")``).
    - A factory that uses ``accelerate.init_empty_weights()``.
    - A factory that loads a lighter config and only materializes the
      module skeleton.

    ``skeleton_factory`` is ignored for trainable stores; trainable models
    always reuse the canonical instance so optimizer parameter identity
    stays stable, and concurrent bindings are still rejected.
    """

    def __init__(
        self,
        *,
        key: str,
        estimated_cache_bytes: int,
        factory: Callable[[], M],
        skeleton_factory: Callable[[], M] | None = None,
        blocks_attr: list[str] = [],  # noqa: B006  (read-only; never mutated)
        stream_trainable_weights: bool = False,
    ) -> None:
        def store_factory() -> ResourceStore:
            model = factory()
            return ModelOffloaderStore.from_module(
                model,
                blocks_attr=blocks_attr,
                stream_trainable_weights=stream_trainable_weights,
            )

        def bind(store: ResourceStore) -> ResourceBinding[M]:
            offloader_store = cast(ModelOffloaderStore, store)
            if skeleton_factory is None or offloader_store.has_trainables:
                # The canonical model is factory()'s output; the store
                # only knows it as nn.Module.
                model = cast(M, offloader_store.model)
            else:
                model = skeleton_factory()
            binding = offloader_store.bind(model)
            # ModelOffloader.value is typed nn.Module but yields the
            # bound module, which is an M.
            return cast(ResourceBinding[M], binding)

        def allow_concurrent_binding(store: ResourceStore) -> bool:
            if skeleton_factory is None:
                return False
            return not cast(ModelOffloaderStore, store).has_trainables

        super().__init__(
            key=key,
            estimated_cache_bytes=estimated_cache_bytes,
            store_factory=store_factory,
            bind=bind,
            allow_concurrent_binding=allow_concurrent_binding,
        )


class LoRASpec(ResourceSpec[LoRA]):
    """LoRA-specific cache spec built from a state-dict factory."""

    def __init__(
        self,
        *,
        key: str,
        estimated_cache_bytes: int,
        factory: Callable[[], dict[str, torch.Tensor]],
    ) -> None:
        def store_factory() -> ResourceStore:
            return LoRA(factory())

        def bind(store: ResourceStore) -> ResourceBinding[LoRA]:
            return cast(LoRA, store).bind()

        super().__init__(
            key=key,
            estimated_cache_bytes=estimated_cache_bytes,
            store_factory=store_factory,
            bind=bind,
        )


class _ObjectStore(Generic[T]):
    """Holds one plain Python object for :class:`ObjectSpec`.

    Satisfies both :class:`ResourceStore` and :class:`ResourceBinding`
    shapes: the store is its own binding because there is no per-use
    state -- ``activate``/``deactivate`` are lifecycle no-ops and the
    ``device`` argument is ignored.
    """

    def __init__(self, value: T, cache_bytes: int) -> None:
        self._value = value
        self._cache_bytes = cache_bytes

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes

    @property
    def value(self) -> T:
        return self._value

    def activate(
        self, device: torch.device | str | None = None, **kwargs: object,
    ) -> None:
        pass

    def deactivate(self) -> None:
        pass


class ObjectSpec(ResourceSpec[T]):
    """Cache spec for a general Python object (tokenizer, processor,
    config, ...) that is not a tensor resource.

    ``factory`` runs once on cache miss; every ``use()`` then yields the
    *same* object instance, including concurrent bindings. Treat the
    object as read-only while bound -- mutating shared state (e.g. a
    tokenizer's padding side) from concurrent uses is a caller-side race.
    The ``device`` argument of ``use()`` is ignored.

    ``estimated_cache_bytes`` defaults to 0 because plain heap objects do
    not compete with the pinned-host-memory budget the cache enforces.
    Zero-byte entries are never useful eviction victims (the default LRU
    policy skips them), so they stay cached until explicitly evicted,
    unregistered, or replaced. Pass a positive estimate only for objects
    whose memory should genuinely count against the budget; it is charged
    as-is since plain objects cannot report actual bytes.
    """

    def __init__(
        self,
        *,
        key: str,
        factory: Callable[[], T],
        estimated_cache_bytes: int = 0,
    ) -> None:
        def store_factory() -> ResourceStore:
            return _ObjectStore(factory(), estimated_cache_bytes)

        def bind(store: ResourceStore) -> ResourceBinding[T]:
            return cast(_ObjectStore[T], store)

        super().__init__(
            key=key,
            estimated_cache_bytes=estimated_cache_bytes,
            store_factory=store_factory,
            bind=bind,
        )


LoRARef: TypeAlias = str | LoRASpec


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Per-key state returned by :meth:`ModelCache.info`."""

    key: str
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
        # Zero-byte entries (e.g. ObjectSpec with the default estimate)
        # free nothing, so evicting them can never help admit an entry.
        candidate_bytes = {
            candidate.key: candidate.cache_bytes
            for candidate in context.candidates
            if candidate.cache_bytes > 0
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
    spec: ResourceSpec
    store: ResourceStore | None = None
    cache_bytes: int = 0  # actual, post-build
    bindings: list[ResourceBinding[Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ModelCache
# ---------------------------------------------------------------------------


class ModelCache:
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
        """Current signed cache-budget headroom.

        Negative means a cached store grew after admission and the
        cache is currently over budget.
        """
        with self._lock:
            return self._available_cache_bytes

    @property
    def _available_cache_bytes(self) -> int:
        return self._max_cache_bytes - self._used_bytes

    def register(self, spec: ResourceSpec, *, replace: bool = False) -> None:
        """Register a lazy store factory without building it.

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
                if existing.bindings:
                    raise ModelInUseError(
                        f"cannot replace active registration {spec.key!r} "
                        f"(bindings={len(existing.bindings)})"
                    )
                if existing.store is not None:
                    self._evict_inactive(spec.key)
            self._entries[spec.key] = _Entry(spec=spec)

    def unregister(self, key: str, *, evict: bool = True) -> None:
        """Drop a registration. If a built store exists and
        ``evict=True`` (default), evict it; otherwise raise
        :class:`ModelInUseError` if it's cached or active."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            if entry.bindings:
                raise ModelInUseError(
                    f"cannot unregister active model {key!r} "
                    f"(bindings={len(entry.bindings)})"
                )
            if entry.store is not None:
                if not evict:
                    raise ModelInUseError(
                        f"{key!r} has a built store; pass evict=True to release it"
                    )
                self._evict_inactive(key)
            del self._entries[key]

    @contextlib.contextmanager
    def use(
        self,
        model: str | ResourceSpec[T],
        *,
        device: torch.device | str | None = None,
        num_resident_blocks: int = 1,
        num_prefetch_blocks: int = 2,
        cyclic: bool = False,
        loras: Sequence[LoRARef] | None = None,
        lora_strengths: Sequence[float] | None = None,
        lora_mode: LoraMode = "merge",
    ) -> Iterator[T]:
        """Acquire an active binding on a cached resource.

        Accepts either a registered key (string) or a :class:`ResourceSpec`
        (auto-registers if its key isn't already known). Yields the
        resource binding's :attr:`~ResourceBinding.value` for use; on context
        exit, releases the binding.

        ``device`` optionally selects the activation device for this
        acquire. It is normalized and passed to the binding's
        :meth:`~ResourceBinding.activate` for that binding. Same-key
        nested acquires create independent bindings when the spec opts
        in (see :attr:`ResourceSpec.allow_concurrent_binding`); specs
        without that opt-in — including the default :class:`ModelSpec`
        without a ``skeleton_factory`` and all trainable model specs —
        reject the second concurrent binding. Different keys and any
        opted-in multiple bindings for the same key may be active on the
        same device at the same time; caller code owns GPU memory
        planning.

        ``loras`` optionally selects cached LoRA resources to apply to
        a :class:`ModelOffloader` binding before activation.
        ``lora_strengths`` defaults to ``1.0`` per LoRA.
        """
        lora_entries: list[_Entry] = []
        lora_bindings: list[ResourceBinding[Any]] = []
        with self._lock:
            for lora in self._lora_refs(loras):
                lora_entry, lora_binding = self._prepare_binding(lora)
                lora_entries.append(lora_entry)
                lora_bindings.append(lora_binding)
            entry, binding = self._prepare_binding(model)
            self._set_loras(
                binding,
                lora_bindings,
                strengths=lora_strengths,
                mode=lora_mode,
            )
            self._activate_binding(
                entry,
                binding,
                device=device,
                stream_config=StreamConfig(
                    num_resident_blocks=num_resident_blocks,
                    num_prefetch_blocks=num_prefetch_blocks,
                    cyclic=cyclic,
                ),
            )
            for lora_entry, lora_binding in zip(
                lora_entries, lora_bindings, strict=True,
            ):
                self._activate_binding(lora_entry, lora_binding)
        try:
            yield binding.value
        finally:
            with self._lock:
                self._release(entry, binding)
                self._release_bindings(lora_entries, lora_bindings)

    def evict(self, key: str) -> None:
        """Manually evict one inactive cached entry. Raises
        :class:`ModelInUseError` if active, no-op if not cached."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.store is None:
                return
            if entry.bindings:
                raise ModelInUseError(f"cannot evict active model {key!r}")
            self._evict_inactive(key)

    def clear(self) -> None:
        """Evict all inactive entries. Registrations are preserved.
        Raises :class:`ModelInUseError` if any entry is active."""
        with self._lock:
            active = [k for k, e in self._entries.items() if e.bindings]
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
                estimated_cache_bytes=entry.spec.estimated_cache_bytes,
                cache_bytes=entry.cache_bytes if entry.store is not None else None,
                cached=entry.store is not None,
                active_count=len(entry.bindings),
            )

    # ------------------------------------------------------------------
    # Binding lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_device(device: torch.device | str | None) -> torch.device | None:
        return canonical_device(device) if device is not None else None

    def _get_entry(self, resource: str | ResourceSpec) -> _Entry:
        """Resolve a resource key/spec to its cache entry, auto-registering
        specs that are not known yet."""
        spec = resource if isinstance(resource, ResourceSpec) else None
        key = spec.key if spec is not None else resource
        assert isinstance(key, str)

        entry = self._entries.get(key)
        if entry is not None:
            return entry
        if spec is None:
            raise ModelNotRegisteredError(
                f"{key!r} is not registered; pass a ResourceSpec to use() or call register() first"
            )
        self.register(spec)
        return self._entries[key]

    def _ensure_store(self, entry: _Entry) -> None:
        """Build and cache the store if this is a cache miss."""
        if entry.store is None:
            self._build_into_entry(entry)

    def _prepare_binding(
        self,
        resource: str | ResourceSpec,
    ) -> tuple[_Entry, ResourceBinding[Any]]:
        """Resolve a resource, ensure its store exists, and create an
        unpublished binding."""
        entry = self._get_entry(resource)
        self._ensure_store(entry)
        return entry, self._create_binding(entry)

    @staticmethod
    def _lora_refs(loras: Sequence[LoRARef] | None) -> Sequence[LoRARef]:
        if loras is None:
            return ()
        if isinstance(loras, str):
            raise TypeError(
                "loras must be a sequence of LoRA keys/specs, not a string"
        )
        return loras

    @staticmethod
    def _set_loras(
        model_binding: ResourceBinding[Any],
        lora_bindings: Sequence[ResourceBinding[Any]],
        *,
        strengths: Sequence[float] | None,
        mode: LoraMode,
    ) -> None:
        if not lora_bindings:
            return
        if not isinstance(model_binding, ModelOffloader):
            raise TypeError(
                "LoRAs can only be applied to ModelOffloader bindings"
            )
        model_binding.set_loras(
            [binding.value for binding in lora_bindings],
            strengths=strengths,
            mode=mode,
        )

    def _create_binding(self, entry: _Entry) -> ResourceBinding[Any]:
        """Create an unpublished binding from the cached store."""
        store = entry.store
        assert store is not None
        if entry.bindings and not entry.spec.allow_concurrent_binding(store):
            raise ModelInUseError(
                f"cannot create multiple active bindings for resource "
                f"{entry.spec.key!r}; spec does not allow concurrent bindings"
            )
        return entry.spec.bind(store)

    def _activate_binding(
        self,
        entry: _Entry,
        binding: ResourceBinding[Any],
        *,
        device: torch.device | str | None = None,
        stream_config: StreamConfig | None = None,
    ) -> None:
        """Activate and publish a binding after all inactive binding
        configuration has been applied."""
        active_device = self._normalize_device(device)
        binding.activate(active_device, stream_config=stream_config)
        entry.bindings.append(binding)
        self._eviction.mark_active(entry.spec.key)

    def _release_bindings(
        self,
        entries: Sequence[_Entry],
        bindings: Sequence[ResourceBinding[Any]],
    ) -> None:
        for entry, binding in reversed(list(zip(entries, bindings, strict=True))):
            self._release(entry, binding)

    def _release(self, entry: _Entry, binding: ResourceBinding[Any]) -> None:
        """End a binding use. On the final release, deactivate and mark the
        entry inactive for eviction policy state. A raising deactivate
        leaves the binding unrecoverable: discard it and propagate so
        the caller sees the binding's failure."""
        try:
            binding.deactivate()
        except BaseException:
            self._discard_binding(entry, binding)
            raise
        self._discard_binding(entry, binding)

    def _discard_binding(
        self,
        entry: _Entry,
        target_binding: ResourceBinding[Any],
    ) -> None:
        for index, binding in enumerate(entry.bindings):
            if binding is target_binding:
                del entry.bindings[index]
                break
        self._mark_inactive_if_idle(entry)

    def _mark_inactive_if_idle(self, entry: _Entry) -> None:
        if entry.store is not None and not entry.bindings:
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
        drop the local store ref BEFORE the host-cache flush so
        refcount-GC frees pinned tensors in time for the flush to
        actually reclaim them.
        """
        estimate = entry.spec.estimated_cache_bytes
        self._evict_to_fit(estimate)
        store: ResourceStore | None = entry.spec.store_factory()

        try:
            actual = store.cache_bytes
            if actual < 0:
                raise ModelCacheError(
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
        # Initial commit: cache_bytes is currently 0; reconcile against
        # actual to record the bytes and update peak. Build-path
        # eviction already made room, so the over-budget warning path
        # is unreachable from here.
        self._reconcile_bytes(entry, actual, phase="build")
        self._eviction.mark_inactive(entry.spec.key)

    def _reconcile_bytes(self, entry: _Entry, observed: int, *, phase: str) -> None:
        """Apply ``observed`` to entry and global byte accounting.

        Used on the build path (initial commit, 0 -> actual).
        Raises :class:`ModelCacheError` if ``observed < 0`` so a
        misbehaving store can't corrupt ``_used_bytes`` (the build
        path also pre-validates before attaching to keep refcount-GC
        timing correct on the cleanup path). Logs a warning if the
        update pushes total usage over budget.
        """
        if observed < 0:
            raise ModelCacheError(
                f"store.cache_bytes for {entry.spec.key!r} returned {observed} (must be >= 0) after {phase}"
            )
        if observed == entry.cache_bytes:
            return
        delta = observed - entry.cache_bytes
        entry.cache_bytes = observed
        self._used_bytes += delta
        if self._used_bytes > self._max_cache_bytes:
            logger.warning(
                "ModelCache over budget after %s: %r grew to %d bytes; total %d/%d. "
                "The store reported a smaller cache_bytes earlier than at this "
                "point -- stores should make cache_bytes final at admission.",
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
            if entry.store is None or entry.bindings:
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
        assert entry.store is not None
        assert not entry.bindings
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
