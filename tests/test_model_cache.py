"""Tests for ``torch_offload.model_cache.ModelCache``.

Uses a fake :class:`ModelStrategy` (``FakeStrategy``) to exercise the
cache without needing actual GPU memory or pinning. The fake records
every lifecycle call so tests can assert ordering and counts.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

import pytest
import torch
from torch import nn

from torch_offload import (
    ActivationError,
    DuplicateModelKeyError,
    EvictionContext,
    EvictionPolicy,
    EvictionPolicyError,
    GpuDeviceOccupiedError,
    ModelCache,
    ModelCacheError,
    ModelInUseError,
    ModelNotRegisteredError,
    ModelSpec,
    ModelTooLargeError,
    PlacementLease,
    PlacementPolicy,
)
from torch_offload.protocols import ModelStrategy

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Fake strategy
# ---------------------------------------------------------------------------


class FakeStrategy:
    """In-memory fake satisfying the :class:`ModelStrategy` protocol.

    Records every lifecycle call. Optionally raises on activate via
    an injected exception for failure-mode tests.
    """

    instances: list["FakeStrategy"] = []  # populated by FakeStrategy() constructor

    def __init__(
        self,
        cache_bytes: int,
        *,
        activate_raises: BaseException | None = None,
    ) -> None:
        self._cache_bytes = cache_bytes
        self._activate_raises = activate_raises
        self._active = False
        self.events: list[str] = []
        self.activate_devices: list[torch.device | None] = []
        self.module = nn.Identity()
        FakeStrategy.instances.append(self)

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes

    @property
    def model(self) -> nn.Module:
        return self.module

    @property
    def value(self) -> nn.Module:
        return self.module

    def activate(self, device: torch.device | str | None = None) -> None:
        self.events.append("activate")
        self.activate_devices.append(torch.device(device) if device is not None else None)
        if self._activate_raises is not None:
            raise self._activate_raises
        if self._active:
            raise RuntimeError("FakeStrategy already active")
        self._active = True

    def deactivate(self) -> None:
        self.events.append("deactivate")
        self._active = False

    def __enter__(self) -> nn.Module:
        self.activate()
        return self.model

    def __exit__(self, *exc) -> None:
        self.deactivate()


def _make_factory(
    bytes_: int,
    *,
    activate_raises: BaseException | None = None,
    factory_raises: BaseException | None = None,
) -> Callable[[], FakeStrategy]:
    def factory() -> FakeStrategy:
        if factory_raises is not None:
            raise factory_raises
        return FakeStrategy(bytes_, activate_raises=activate_raises)

    return factory


def _spec(key: str, bytes_: int, **kwargs) -> ModelSpec:
    return ModelSpec(key=key, estimated_cache_bytes=bytes_, factory=_make_factory(bytes_, **kwargs))


def _is_registered(cache: ModelCache, key: str) -> bool:
    try:
        cache.info(key)
    except ModelNotRegisteredError:
        return False
    return True


def _is_cached(cache: ModelCache, key: str) -> bool:
    return cache.info(key).cached


def _active_refcounts(cache: ModelCache, *keys: str) -> dict[str, int]:
    return {key: info.active_count for key in keys if (info := cache.info(key)).active_count > 0}


class MRUEvictionPolicy:
    """Test policy that evicts the most-recently inactive key first."""

    def __init__(self) -> None:
        self.inactive: list[str] = []

    def mark_active(self, key: str) -> None:
        self.discard(key)

    def mark_inactive(self, key: str) -> None:
        self.discard(key)
        self.inactive.append(key)

    def discard(self, key: str) -> None:
        self.inactive = [existing for existing in self.inactive if existing != key]

    def choose_victims(self, context: EvictionContext) -> tuple[str, ...]:
        if context.bytes_to_free <= 0:
            return ()
        chosen: list[str] = []
        freed = 0
        candidate_bytes = {candidate.key: candidate.cache_bytes for candidate in context.candidates}
        ordered = [key for key in self.inactive if key in candidate_bytes]
        ordered_set = set(ordered)
        ordered.extend(candidate.key for candidate in context.candidates if candidate.key not in ordered_set)
        for key in reversed(ordered):
            chosen.append(key)
            freed += candidate_bytes[key]
            if freed >= context.bytes_to_free:
                break
        return tuple(chosen)


class LargestFirstEvictionPolicy:
    """Test policy that uses candidate sizes from EvictionContext."""

    def mark_active(self, key: str) -> None:
        del key

    def mark_inactive(self, key: str) -> None:
        del key

    def discard(self, key: str) -> None:
        del key

    def choose_victims(self, context: EvictionContext) -> tuple[str, ...]:
        if context.bytes_to_free <= 0:
            return ()
        ordered = sorted(
            context.candidates,
            key=lambda candidate: candidate.cache_bytes,
            reverse=True,
        )
        chosen: list[str] = []
        freed = 0
        for candidate in ordered:
            chosen.append(candidate.key)
            freed += candidate.cache_bytes
            if freed >= context.bytes_to_free:
                break
        return tuple(chosen)


class BadEvictionPolicy(MRUEvictionPolicy):
    """Test policy that returns an invalid victim."""

    def choose_victims(self, context: EvictionContext) -> tuple[str, ...]:
        del context
        return ("missing",)


class DuplicateVictimEvictionPolicy(MRUEvictionPolicy):
    """Test policy that returns the same victim more than once."""

    def choose_victims(self, context: EvictionContext) -> tuple[str, ...]:
        return (context.candidates[0].key, context.candidates[0].key)


class UnderSelectingEvictionPolicy(MRUEvictionPolicy):
    """Test policy that returns valid but insufficient victims."""

    def choose_victims(self, context: EvictionContext) -> tuple[str, ...]:
        return tuple(candidate.key for candidate in context.candidates[:1])


class SharingPlacementPolicy:
    """Test policy that allows different keys to share a CUDA device."""

    def reserve(self, *, key: str, requested_device: torch.device | str | None) -> PlacementLease:
        device = torch.device(requested_device) if requested_device is not None else None
        return PlacementLease(key=key, device=device)

    def validate_reentrant(
        self,
        *,
        key: str,
        active_device: torch.device | None,
        requested_device: torch.device | str | None,
    ) -> None:
        device = torch.device(requested_device) if requested_device is not None else None
        if device is not None and active_device != device:
            raise ModelCacheError(f"{key!r} active on {active_device}, requested {device}")

    def release(self, lease: PlacementLease) -> None:
        del lease


@pytest.fixture(autouse=True)
def _clear_instances():
    FakeStrategy.instances.clear()
    yield
    FakeStrategy.instances.clear()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_negative_max_bytes_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_cache_bytes"):
            ModelCache(-1)

    def test_zero_max_bytes_allowed_for_zero_byte_strategies(self) -> None:
        # Edge case: a degenerate cache that only accepts zero-byte
        # handles (always-on-GPU passthrough, mmap-only, etc.).
        cache = ModelCache(0)
        assert cache.max_cache_bytes == 0
        assert cache.used_cache_bytes == 0
        assert cache.available_cache_bytes == 0


# ---------------------------------------------------------------------------
# Hit / miss / build
# ---------------------------------------------------------------------------


class TestHitMiss:
    def test_first_use_builds_via_factory(self) -> None:
        cache = ModelCache(1000)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        assert len(FakeStrategy.instances) == 1
        assert cache.info("a").cached
        assert cache.used_cache_bytes == 100
        assert cache.available_cache_bytes == 900

    def test_second_use_hits_cache(self) -> None:
        cache = ModelCache(1000)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        strategy = FakeStrategy.instances[0]
        with cache.use("a"):
            pass
        assert len(FakeStrategy.instances) == 1
        assert strategy.events == ["activate", "deactivate", "activate", "deactivate"]

    def test_use_with_spec_auto_registers(self) -> None:
        cache = ModelCache(1000)
        with cache.use(_spec("a", 100)):
            pass
        assert _is_registered(cache, "a")

    def test_use_with_string_key_requires_registration(self) -> None:
        cache = ModelCache(1000)
        with pytest.raises(ModelNotRegisteredError):
            with cache.use("missing"):
                pass


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_duplicate_key_raises_without_replace(self) -> None:
        cache = ModelCache(1000)
        cache.register(_spec("a", 100))
        with pytest.raises(DuplicateModelKeyError):
            cache.register(_spec("a", 200))

    def test_replace_evicts_existing_inactive(self) -> None:
        cache = ModelCache(1000)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        cache.register(_spec("a", 200), replace=True)
        # The freshly-replaced entry has not been built yet.
        assert cache.used_cache_bytes == 0
        # The previous strategy was released during replace.

    def test_replace_active_raises(self) -> None:
        cache = ModelCache(1000)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            with pytest.raises(ModelInUseError):
                cache.register(_spec("a", 200), replace=True)

    def test_unregister_evicts_by_default(self) -> None:
        cache = ModelCache(1000)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        cache.unregister("a")
        assert not _is_registered(cache, "a")

    def test_unregister_active_raises(self) -> None:
        cache = ModelCache(1000)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            with pytest.raises(ModelInUseError):
                cache.unregister("a")

    def test_unregister_unknown_key_is_noop(self) -> None:
        cache = ModelCache(1000)
        cache.unregister("missing")  # no error


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


class TestEviction:
    def test_lru_evicts_oldest_when_full(self) -> None:
        cache = ModelCache(300)
        for k in ["a", "b", "c"]:
            cache.register(_spec(k, 100))
        for k in ["a", "b", "c"]:
            with cache.use(k):
                pass
        # Cache is full at 300/300; adding "d" must evict "a" (LRU).
        cache.register(_spec("d", 100))
        with cache.use("d"):
            pass
        assert not _is_cached(cache, "a")
        assert _is_cached(cache, "b")
        assert _is_cached(cache, "c")
        assert _is_cached(cache, "d")

    def test_multi_evict_for_one_admit(self) -> None:
        cache = ModelCache(300)
        for k in ["a", "b", "c"]:
            cache.register(_spec(k, 100))
            with cache.use(k):
                pass
        # Adding a 250-byte model needs 3 evictions of 100-byte models
        # to make room (250 > 300 - 100 - 100).
        cache.register(_spec("big", 250))
        with cache.use("big"):
            pass
        assert cache.used_cache_bytes == 250
        assert not _is_cached(cache, "a")
        assert not _is_cached(cache, "b")
        assert not _is_cached(cache, "c")
        assert _is_cached(cache, "big")

    def test_custom_eviction_policy_controls_victim_selection(self) -> None:
        policy: EvictionPolicy = MRUEvictionPolicy()
        cache = ModelCache(200, eviction_policy=policy)
        cache.register(_spec("a", 100))
        cache.register(_spec("b", 100))
        with cache.use("a"):
            pass
        with cache.use("b"):
            pass

        # MRU policy evicts "b" rather than the default LRU victim "a".
        cache.register(_spec("c", 100))
        with cache.use("c"):
            pass

        assert _is_cached(cache, "a")
        assert not _is_cached(cache, "b")
        assert _is_cached(cache, "c")

    def test_custom_eviction_policy_can_use_candidate_sizes(self) -> None:
        policy: EvictionPolicy = LargestFirstEvictionPolicy()
        cache = ModelCache(200, eviction_policy=policy)
        cache.register(_spec("small", 80))
        cache.register(_spec("large", 120))
        with cache.use("small"):
            pass
        with cache.use("large"):
            pass

        cache.register(_spec("incoming", 100))
        with cache.use("incoming"):
            pass

        assert _is_cached(cache, "small")
        assert not _is_cached(cache, "large")
        assert _is_cached(cache, "incoming")
        assert cache.used_cache_bytes == 180

    def test_eviction_policy_cannot_choose_non_candidate(self) -> None:
        policy: EvictionPolicy = BadEvictionPolicy()
        cache = ModelCache(200, eviction_policy=policy)
        cache.register(_spec("a", 100))
        cache.register(_spec("b", 100))
        with cache.use("a"):
            pass
        with cache.use("b"):
            pass

        with pytest.raises(EvictionPolicyError, match="invalid victims"):
            with cache.use(_spec("c", 100)):
                pass

    def test_eviction_policy_cannot_choose_duplicate_victims(self) -> None:
        policy: EvictionPolicy = DuplicateVictimEvictionPolicy()
        cache = ModelCache(200, eviction_policy=policy)
        cache.register(_spec("a", 100))
        cache.register(_spec("b", 100))
        with cache.use("a"):
            pass
        with cache.use("b"):
            pass

        with pytest.raises(EvictionPolicyError, match="invalid victims"):
            with cache.use(_spec("c", 100)):
                pass

    def test_eviction_policy_must_choose_enough_victims(self) -> None:
        policy: EvictionPolicy = UnderSelectingEvictionPolicy()
        cache = ModelCache(300, eviction_policy=policy)
        for key in ("a", "b", "c"):
            cache.register(_spec(key, 100))
            with cache.use(key):
                pass

        with pytest.raises(EvictionPolicyError, match="insufficient victims"):
            with cache.use(_spec("big", 250)):
                pass

        assert cache.used_cache_bytes == 300
        assert all(_is_cached(cache, key) for key in ("a", "b", "c"))

    def test_too_large_raises(self) -> None:
        cache = ModelCache(100)
        with pytest.raises(ModelTooLargeError) as excinfo:
            with cache.use(_spec("oversized", 200)):
                pass
        err = excinfo.value
        assert err.required == 200
        assert err.limit == 100

    def test_too_large_due_to_active_blockers_raises(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            # "a" is active and uses 100/200; "a" cannot be evicted.
            # Trying to admit a 150-byte "b" needs 50 more bytes than
            # currently free, but the only entry that could be evicted
            # is the active one.
            with pytest.raises(ModelTooLargeError) as excinfo:
                with cache.use(_spec("b", 150)):
                    pass
            err = excinfo.value
            assert err.required == 150
            assert err.used == 100
            assert err.limit == 200

    def test_active_entries_are_not_evicted(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        cache.register(_spec("b", 100))
        with cache.use("a"):
            with cache.use("b"):
                pass
        cache.register(_spec("c", 100))
        with cache.use("c"):
            pass
        assert _is_cached(cache, "a")
        assert not _is_cached(cache, "b")
        assert _is_cached(cache, "c")

    def test_manual_evict(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        cache.evict("a")
        assert cache.used_cache_bytes == 0

    def test_evict_active_raises(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            with pytest.raises(ModelInUseError):
                cache.evict("a")

    def test_evict_unknown_is_noop(self) -> None:
        cache = ModelCache(200)
        cache.evict("missing")
        cache.register(_spec("a", 100))
        cache.evict("a")  # registered but never built

    def test_clear_evicts_all_inactive(self) -> None:
        cache = ModelCache(300)
        for k in ["a", "b", "c"]:
            cache.register(_spec(k, 100))
            with cache.use(k):
                pass
        cache.clear()
        assert cache.used_cache_bytes == 0
        # Registrations preserved.
        assert all(_is_registered(cache, key) for key in ("a", "b", "c"))
        assert not any(_is_cached(cache, key) for key in ("a", "b", "c"))

    def test_clear_with_active_raises(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            with pytest.raises(ModelInUseError):
                cache.clear()


# ---------------------------------------------------------------------------
# Active-set refcount semantics
# ---------------------------------------------------------------------------


class TestActiveSet:
    def test_nested_same_key_refcounts(self) -> None:
        # Re-entrant acquire on the same key bumps refcount but does
        # NOT call the strategy's activate again.
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        with cache.use("a") as m1:
            with cache.use("a") as m2:
                # Same module yielded both times.
                assert m1 is m2
        s = FakeStrategy.instances[0]
        # activate called once, deactivate once.
        assert s.events.count("activate") == 1
        assert s.events.count("deactivate") == 1

    def test_concurrent_different_keys(self) -> None:
        # Encoder + decoder co-resident — both active simultaneously,
        # both protected from eviction.
        cache = ModelCache(300)
        cache.register(_spec("enc", 100))
        cache.register(_spec("dec", 100))
        with cache.use("enc") as e:
            with cache.use("dec") as d:
                assert isinstance(e, nn.Module)
                assert isinstance(d, nn.Module)
                assert _active_refcounts(cache, "enc", "dec") == {"enc": 1, "dec": 1}

    def test_deactivate_returns_to_lru_at_mru(self) -> None:
        cache = ModelCache(300)
        for k in ["a", "b", "c"]:
            cache.register(_spec(k, 100))
            with cache.use(k):
                pass
        # Touching "a" makes it MRU, so admitting "d" evicts "b".
        with cache.use("a"):
            pass
        cache.register(_spec("d", 100))
        with cache.use("d"):
            pass
        assert _is_cached(cache, "a")
        assert not _is_cached(cache, "b")
        assert _is_cached(cache, "c")
        assert _is_cached(cache, "d")


# ---------------------------------------------------------------------------
# Naive GPU placement + thread-safe cache leases
# ---------------------------------------------------------------------------


class TestGpuPlacementAndConcurrency:
    def test_different_cuda_devices_can_be_active_together(self) -> None:
        cache = ModelCache(300)
        cache.register(_spec("a", 100))
        cache.register(_spec("b", 100))

        with cache.use("a", device="cuda:0"):
            with cache.use("b", device="cuda:1"):
                assert _active_refcounts(cache, "a", "b") == {"a": 1, "b": 1}

        assert FakeStrategy.instances[0].activate_devices == [torch.device("cuda:0")]
        assert FakeStrategy.instances[1].activate_devices == [torch.device("cuda:1")]

    def test_different_key_same_cuda_device_is_rejected_before_build(self) -> None:
        cache = ModelCache(300)
        cache.register(_spec("a", 100))
        cache.register(_spec("b", 100))

        with cache.use("a", device="cuda:0"):
            with pytest.raises(GpuDeviceOccupiedError) as excinfo:
                with cache.use("b", device="cuda:0"):
                    pass

        err = excinfo.value
        assert err.device == torch.device("cuda:0")
        assert err.key == "b"
        assert err.active_key == "a"
        # The rejected key never built its strategy.
        assert len(FakeStrategy.instances) == 1

    def test_custom_placement_policy_can_allow_cuda_device_sharing(self) -> None:
        policy: PlacementPolicy = SharingPlacementPolicy()
        cache = ModelCache(300, placement_policy=policy)
        cache.register(_spec("a", 100))
        cache.register(_spec("b", 100))

        with cache.use("a", device="cuda:0"):
            with cache.use("b", device="cuda:0"):
                assert _active_refcounts(cache, "a", "b") == {"a": 1, "b": 1}

        assert FakeStrategy.instances[0].activate_devices == [torch.device("cuda:0")]
        assert FakeStrategy.instances[1].activate_devices == [torch.device("cuda:0")]

    def test_final_release_frees_cuda_device_for_another_key(self) -> None:
        cache = ModelCache(300)
        cache.register(_spec("a", 100))
        cache.register(_spec("b", 100))

        with cache.use("a", device="cuda:0"):
            pass
        with cache.use("b", device="cuda:0"):
            pass

        assert FakeStrategy.instances[0].activate_devices == [torch.device("cuda:0")]
        assert FakeStrategy.instances[1].activate_devices == [torch.device("cuda:0")]

    def test_activation_failure_releases_cuda_device(self) -> None:
        cache = ModelCache(300)
        cache.register(_spec("a", 100, activate_raises=RuntimeError("gpu boom")))
        cache.register(_spec("b", 100))

        with pytest.raises(ActivationError):
            with cache.use("a", device="cuda:0"):
                pass

        with cache.use("b", device="cuda:0"):
            pass

        assert FakeStrategy.instances[1].activate_devices == [torch.device("cuda:0")]

    def test_different_keys_on_different_cuda_devices_can_overlap_across_threads(self) -> None:
        cache = ModelCache(300)
        cache.register(_spec("a", 100))
        cache.register(_spec("b", 100))

        entered = threading.Barrier(3)
        release = threading.Event()
        errors = queue.SimpleQueue()

        def worker(key: str, device: str) -> None:
            try:
                with cache.use(key, device=device):
                    entered.wait(timeout=2)
                    if not release.wait(timeout=2):
                        raise TimeoutError("release event was not set")
            except BaseException as exc:
                errors.put(exc)

        threads = [
            threading.Thread(target=worker, args=("a", "cuda:0")),
            threading.Thread(target=worker, args=("b", "cuda:1")),
        ]
        for thread in threads:
            thread.start()

        try:
            entered.wait(timeout=2)
            assert _active_refcounts(cache, "a", "b") == {"a": 1, "b": 1}
        finally:
            release.set()
            for thread in threads:
                thread.join(timeout=2)

        assert all(not thread.is_alive() for thread in threads)
        captured = []
        while not errors.empty():
            captured.append(errors.get())
        assert captured == []


# ---------------------------------------------------------------------------
# Acquire-time device selection
# ---------------------------------------------------------------------------


class TestDeviceSelection:
    def test_use_passes_device_to_activate(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))

        with cache.use("a", device="cpu") as m:
            assert isinstance(m, nn.Module)

        s = FakeStrategy.instances[0]
        assert s.activate_devices == [torch.device("cpu")]

    def test_reentrant_same_device_allowed(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))

        with cache.use("a", device="cpu"):
            with cache.use("a", device=torch.device("cpu")):
                pass

        s = FakeStrategy.instances[0]
        assert s.events.count("activate") == 1
        assert s.events.count("deactivate") == 1

    def test_reentrant_indexed_cpu_matches_cpu(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))

        with cache.use("a", device="cpu"):
            with cache.use("a", device="cpu:0"):
                pass

        s = FakeStrategy.instances[0]
        assert s.activate_devices == [torch.device("cpu")]

    def test_reentrant_omitted_device_inherits_active_lease(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))

        with cache.use("a", device="cpu"):
            with cache.use("a"):
                pass

        s = FakeStrategy.instances[0]
        assert s.events.count("activate") == 1

    def test_reentrant_different_device_rejected(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))

        with cache.use("a", device="cpu"):
            with pytest.raises(ModelCacheError, match="already active"):
                with cache.use("a", device="meta"):
                    pass

        s = FakeStrategy.instances[0]
        assert s.events.count("activate") == 1
        assert s.events.count("deactivate") == 1

    def test_reentrant_device_after_unspecified_activation_rejected(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))

        with cache.use("a"):
            with pytest.raises(ModelCacheError, match="without a cache-visible device"):
                with cache.use("a", device="cpu"):
                    pass

    def test_inactive_entry_can_reactivate_on_different_device(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))

        with cache.use("a", device="cpu"):
            pass
        with cache.use("a", device="meta"):
            pass

        s = FakeStrategy.instances[0]
        assert s.activate_devices == [torch.device("cpu"), torch.device("meta")]

    @CUDA
    def test_reentrant_bare_cuda_matches_current_indexed_cuda(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        expected = torch.device("cuda", torch.cuda.current_device())

        with cache.use("a", device="cuda"):
            with cache.use("a", device=expected):
                pass

        s = FakeStrategy.instances[0]
        assert s.activate_devices == [expected]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_factory_failure_leaves_cache_unchanged(self) -> None:
        cache = ModelCache(200)
        spec = ModelSpec(
            key="bad",
            estimated_cache_bytes=100,
            factory=_make_factory(100, factory_raises=RuntimeError("boom")),
        )
        with pytest.raises(RuntimeError, match="boom"):
            with cache.use(spec):
                pass
        assert cache.used_cache_bytes == 0
        # Registration persisted (but no built strategy).
        assert _is_registered(cache, "bad")
        assert not _is_cached(cache, "bad")

    def test_activation_failure_on_fresh_build_drops_entry(self) -> None:
        # Activation failure on a freshly-built strategy: discard it
        # entirely so a retry rebuilds (the failed activation may have
        # left the strategy in an unknown state).
        cache = ModelCache(200)
        spec = ModelSpec(
            key="bad",
            estimated_cache_bytes=100,
            factory=_make_factory(100, activate_raises=RuntimeError("gpu boom")),
        )
        with pytest.raises(ActivationError):
            with cache.use(spec):
                pass
        assert cache.used_cache_bytes == 0
        assert _is_registered(cache, "bad")
        assert not _is_cached(cache, "bad")
        # Strategy was constructed and then released.

    def test_activation_failure_on_cached_entry_discards_it(self) -> None:
        # Activation failure on a previously-cached entry is handled
        # the same as on a freshly-built one. Strategies like
        # block-streaming can fail mid-way through activate after
        # partially installing hooks/pool; caching them as
        # "ready to retry" lies about their state.
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        s = FakeStrategy.instances[0]
        s._activate_raises = RuntimeError("transient gpu")
        with pytest.raises(ActivationError):
            with cache.use("a"):
                pass
        # Entry was discarded — released and removed from cache state.
        assert not _is_cached(cache, "a")
        assert cache.used_cache_bytes == 0
        # The failed strategy reference was dropped.
        # Registration persisted for retry — but next acquire rebuilds.
        assert _is_registered(cache, "a")

    def test_factory_failure_after_pre_eviction_keeps_evictions(self) -> None:
        # Documented behavior: the cache pre-evicts to make room for
        # the factory's pinning. If the factory then fails, those
        # pre-evictions are committed (rolling back would mean
        # re-pinning weights, which can OOM the host allocator).
        # Cache stays internally consistent — used_bytes reflects the
        # post-eviction state, the failed key's registration persists.
        cache = ModelCache(100)
        cache.register(_spec("warm", 100))
        with cache.use("warm"):
            pass
        assert cache.used_cache_bytes == 100
        # Adding a 100-byte "bad" forces eviction of "warm" before
        # the factory runs. Factory then raises.
        bad_spec = ModelSpec(
            key="bad",
            estimated_cache_bytes=100,
            factory=_make_factory(100, factory_raises=RuntimeError("build boom")),
        )
        with pytest.raises(RuntimeError, match="build boom"):
            with cache.use(bad_spec):
                pass
        # warm was evicted to make room — that eviction is committed.
        assert not _is_cached(cache, "warm")
        assert cache.used_cache_bytes == 0
        # bad's registration persists for retry; warm's also.
        assert _is_registered(cache, "warm")
        assert _is_registered(cache, "bad")
        assert not _is_cached(cache, "bad")


# ---------------------------------------------------------------------------
# Deactivate failure
# ---------------------------------------------------------------------------


class TestDeactivateFailure:
    def test_deactivate_failure_discards_entry(self) -> None:
        # A strategy whose deactivate() raises is unrecoverable:
        # the entry is removed from the cache and
        # the original deactivate exception propagates.
        class FailingDeactivateStrategy(FakeStrategy):
            def deactivate(self) -> None:
                self.events.append("deactivate")
                raise RuntimeError("deactivate boom")

        def factory() -> FailingDeactivateStrategy:
            s = FailingDeactivateStrategy(100)
            FakeStrategy.instances.append(s)
            return s

        cache = ModelCache(200)
        cache.register(ModelSpec(key="a", estimated_cache_bytes=100, factory=factory))
        with pytest.raises(RuntimeError, match="deactivate boom"):
            with cache.use("a"):
                pass
        # Entry removed from cache, bytes reclaimed in accounting,
        # registration preserved for retry.
        assert not _is_cached(cache, "a")
        assert cache.used_cache_bytes == 0
        assert _is_registered(cache, "a")
        # The failed strategy reference was dropped.


# ---------------------------------------------------------------------------
# cache_bytes validation
# ---------------------------------------------------------------------------


class TestCacheBytesValidation:
    def test_negative_actual_rejected(self) -> None:
        def factory():
            return FakeStrategy(-1)

        cache = ModelCache(200)
        spec = ModelSpec(key="bad", estimated_cache_bytes=10, factory=factory)
        with pytest.raises(Exception, match="cache_bytes"):
            with cache.use(spec):
                pass
        assert cache.used_cache_bytes == 0
        assert _is_registered(cache, "bad")
        assert not _is_cached(cache, "bad")
        # Strategy was constructed and then released.

    def test_negative_post_activate_rejected(self) -> None:
        # A strategy that reports a sane cache_bytes pre-activate but
        # mutates to negative during activate must be rejected — silently
        # admitting it would corrupt _used_bytes accounting.
        class PostActivateNegative(FakeStrategy):
            def activate(self) -> None:
                super().activate()
                self._cache_bytes = -1

        def factory():
            return PostActivateNegative(100)

        cache = ModelCache(200)
        spec = ModelSpec(key="bad", estimated_cache_bytes=100, factory=factory)
        with pytest.raises(Exception, match="cache_bytes"):
            with cache.use(spec):
                pass
        assert cache.used_cache_bytes == 0
        assert not _is_cached(cache, "bad")


# ---------------------------------------------------------------------------
# Actual-vs-estimate reconciliation
# ---------------------------------------------------------------------------


class TestActualVsEstimate:
    def test_actual_within_estimate_uses_estimate(self) -> None:
        # No reconciliation needed if actual <= estimate.
        cache = ModelCache(200)

        def factory():
            return FakeStrategy(50)  # smaller than estimate

        cache.register(ModelSpec(key="a", estimated_cache_bytes=100, factory=factory))
        with cache.use("a"):
            pass
        # Bytes accounting reflects actual.
        assert cache.used_cache_bytes == 50

    def test_actual_exceeds_estimate_evicts_more(self) -> None:
        cache = ModelCache(300)
        cache.register(_spec("filler", 100))
        with cache.use("filler"):
            pass

        def big_factory():
            return FakeStrategy(250)  # actual is 250 vs estimated 150

        cache.register(
            ModelSpec(key="big", estimated_cache_bytes=150, factory=big_factory),
        )
        with cache.use("big"):
            pass
        # "filler" must have been evicted to make room for the actual 250.
        assert cache.used_cache_bytes == 250
        assert not _is_cached(cache, "filler")

    def test_cache_bytes_reconciled_after_activate(self) -> None:
        # A strategy that reports 0 bytes pre-activate but pins memory
        # during activate (simulating block-streaming with auto_setup=False
        # whose factory forgot to call prepare()) must have its
        # cache_bytes reconciled by the cache after activate so
        # _used_bytes reflects reality.
        class LateBindStrategy(FakeStrategy):
            def __init__(self, *args, late_bytes: int = 100, **kw):
                super().__init__(0, **kw)  # report 0 pre-activate
                self._late_bytes = late_bytes

            def activate(self):
                super().activate()
                # Simulate pinning during activate.
                self._cache_bytes = self._late_bytes

        def factory():
            return LateBindStrategy(late_bytes=100)

        cache = ModelCache(200)
        spec = ModelSpec(key="late", estimated_cache_bytes=10, factory=factory)
        with cache.use(spec):
            # Inside the context, cache_bytes should reflect the
            # post-activate reality (100), not the pre-activate 0.
            info = cache.info("late")
            assert info.cache_bytes == 100
            assert cache.used_cache_bytes == 100

    def test_actual_overflow_rejects_and_releases(self) -> None:
        cache = ModelCache(100)

        def oversized():
            return FakeStrategy(200)  # estimate 50, actual 200

        spec = ModelSpec(key="bad", estimated_cache_bytes=50, factory=oversized)
        with pytest.raises(ModelTooLargeError):
            with cache.use(spec):
                pass
        # The constructed strategy reference was dropped.
        assert cache.used_cache_bytes == 0


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------


class TestObservability:
    def test_info_for_unbuilt_entry(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        info = cache.info("a")
        assert info.key == "a"
        assert info.cached is False
        assert info.cache_bytes is None
        assert info.active_count == 0

    def test_info_for_built_entry(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        info = cache.info("a")
        assert info.cached is True
        assert info.cache_bytes == 100
        assert info.active_count == 0

    def test_info_unknown_raises(self) -> None:
        cache = ModelCache(200)
        with pytest.raises(ModelNotRegisteredError):
            cache.info("missing")

    def test_label_default_is_none(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        assert cache.info("a").label is None

    def test_label_is_passed_through(self) -> None:
        cache = ModelCache(200)
        cache.register(
            ModelSpec(key="a", estimated_cache_bytes=100, factory=_make_factory(100), label="text encoder"),
        )
        assert cache.info("a").label == "text encoder"


# ---------------------------------------------------------------------------
# Host empty cache callback
# ---------------------------------------------------------------------------


class TestHostEmptyCache:
    def test_callback_invoked_after_eviction(self) -> None:
        calls = {"n": 0}
        cache = ModelCache(200, empty_host_cache=lambda: calls.__setitem__("n", calls["n"] + 1))
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        cache.evict("a")
        assert calls["n"] == 1

    def test_callback_not_invoked_on_normal_deactivate(self) -> None:
        calls = {"n": 0}
        cache = ModelCache(200, empty_host_cache=lambda: calls.__setitem__("n", calls["n"] + 1))
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        # No evict yet — callback should not have fired.
        assert calls["n"] == 0

    def test_callback_failure_logged_not_raised(self) -> None:
        def bad():
            raise RuntimeError("flush failed")

        cache = ModelCache(200, empty_host_cache=bad)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        # Should not propagate the callback failure.
        cache.evict("a")


# ---------------------------------------------------------------------------
# pre_activate hook
# ---------------------------------------------------------------------------


class TestPreActivate:
    def test_runs_after_build_before_activate(self) -> None:
        # Hook fires while the strategy is still deactivated. Capture the
        # event ordering by appending into the strategy's events list.
        observed: list[str] = []

        def configure(strategy):
            observed.append(f"pre_activate(active={strategy._active})")

        cache = ModelCache(200)
        with cache.use(_spec("a", 100), pre_activate=configure):
            pass
        s = FakeStrategy.instances[0]
        assert observed == ["pre_activate(active=False)"]
        # pre_activate ran before activate, activate ran before deactivate.
        assert s.events == ["activate", "deactivate"]

    def test_runs_on_cache_hit_too(self) -> None:
        # First use builds + activates; second use hits the cache. The hook
        # must run on both paths (anything keyed off activation state needs
        # per-acquire reconfiguration).
        calls = {"n": 0}

        def configure(strategy):
            calls["n"] += 1

        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        with cache.use("a", pre_activate=configure):
            pass
        with cache.use("a", pre_activate=configure):
            pass
        assert calls["n"] == 2

    def test_failure_discards_entry(self) -> None:
        # A raising hook leaves the strategy in an unknown state — discard
        # the entry, wrap in ActivationError, registration persists for retry.
        def configure(strategy):
            raise RuntimeError("config boom")

        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        with pytest.raises(ActivationError, match="pre_activate"):
            with cache.use("a", pre_activate=configure):
                pass
        assert not _is_cached(cache, "a")
        assert cache.used_cache_bytes == 0
        assert _is_registered(cache, "a")
        # Strategy must NOT have been activated (hook ran before activate
        # and raised, so activate never fired).
        s = FakeStrategy.instances[0]
        assert "activate" not in s.events

    def test_skipped_on_reentrant_use(self) -> None:
        # Re-entrant lease shares the active strategy; the hook would
        # either fail (strategy active, can't reconfigure) or silently
        # violate invariants. Skip it.
        calls = {"n": 0}

        def configure(strategy):
            calls["n"] += 1

        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        with cache.use("a", pre_activate=configure):
            with cache.use("a", pre_activate=configure):
                pass
        assert calls["n"] == 1

    def test_same_key_reentry_from_inside_hook_raises(self) -> None:
        # A hook that re-enters cache.use() for the SAME key would
        # otherwise corrupt eviction-policy inactive state: active_count
        # is still 0 during the hook, so the inner acquire treats it as
        # a fresh lease and runs a full activate/deactivate cycle,
        # leaving the key marked inactive when the outer continues.
        cache = ModelCache(200)
        cache.register(_spec("a", 100))

        def reenter_same(strategy):
            with cache.use("a"):
                pass

        with pytest.raises(ModelCacheError, match="pre_activate"):
            with cache.use("a", pre_activate=reenter_same):
                pass
        # Failure discards the entry just like any pre_activate failure.
        assert not _is_cached(cache, "a")
        assert cache.used_cache_bytes == 0

    def test_different_key_reentry_from_inside_hook_works(self) -> None:
        # Re-entering the cache for a DIFFERENT key during the hook is
        # legitimate: different entries don't share active state.
        cache = ModelCache(300)
        cache.register(_spec("a", 100))
        cache.register(_spec("b", 100))

        captured: list[str] = []

        def use_other(strategy):
            with cache.use("b") as m:
                captured.append(type(m).__name__)

        with cache.use("a", pre_activate=use_other) as m_a:
            assert m_a is not None
        # Both keys remain cached and inactive after the outer lease exits.
        assert _is_cached(cache, "a")
        assert _is_cached(cache, "b")
        assert captured == ["Identity"]

    def test_configuring_flag_clears_on_failure_so_retry_works(self) -> None:
        # If the hook raises, the entry is discarded but the registration
        # persists. A subsequent use() with a non-failing hook must succeed
        # — the configuring flag must be reset on the failure path.
        cache = ModelCache(200)
        cache.register(_spec("a", 100))

        def boom(strategy):
            raise RuntimeError("first attempt fails")

        with pytest.raises(ActivationError):
            with cache.use("a", pre_activate=boom):
                pass

        # Retry without the failing hook — must work.
        with cache.use("a") as m:
            assert m is not None


# ---------------------------------------------------------------------------
# ModelStrategy structural check
# ---------------------------------------------------------------------------


class TestStrategyConformance:
    def test_fake_satisfies_protocol(self) -> None:
        s = FakeStrategy(100)
        assert isinstance(s, ModelStrategy)
