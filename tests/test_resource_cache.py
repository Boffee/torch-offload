"""Tests for the store-only :mod:`torch_offload.resource_cache` API."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from torch_offload import (
    CacheError,
    DuplicateResourceKeyError,
    EvictionContext,
    EvictionPolicyError,
    ResourceCache,
    ObjectSpec,
    ResourceCachedError,
    ResourceLeasedError,
    ResourceNotRegisteredError,
    ResourceSpec,
    ResourceTooLargeError,
)
from torch_offload.protocols import ResourceStore


class FakeStore:
    """In-memory fake satisfying :class:`ResourceStore`."""

    instances: list[FakeStore] = []

    def __init__(self, cache_bytes: int) -> None:
        self._cache_bytes = cache_bytes
        FakeStore.instances.append(self)

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes


@dataclass(frozen=True)
class FakeSpec:
    """Test-only structural :class:`ResourceSpec` implementation."""

    key: str
    estimated_cache_bytes: int
    factory: Callable[[], FakeStore]

    def build_store(self) -> ResourceStore:
        return self.factory()

    def value(self, store: ResourceStore) -> FakeStore:
        assert isinstance(store, FakeStore)
        return store


def _factory(
    cache_bytes: int,
    *,
    raises: BaseException | None = None,
) -> Callable[[], FakeStore]:
    def factory() -> FakeStore:
        if raises is not None:
            raise raises
        return FakeStore(cache_bytes)

    return factory


def _spec(
    key: str,
    cache_bytes: int,
    *,
    estimated_cache_bytes: int | None = None,
    raises: BaseException | None = None,
) -> ResourceSpec[FakeStore]:
    return FakeSpec(
        key=key,
        estimated_cache_bytes=(cache_bytes if estimated_cache_bytes is None else estimated_cache_bytes),
        factory=_factory(cache_bytes, raises=raises),
    )


def _is_registered(cache: ResourceCache, key: str) -> bool:
    try:
        cache.info(key)
    except ResourceNotRegisteredError:
        return False
    return True


def _is_cached(cache: ResourceCache, key: str) -> bool:
    return cache.info(key).cached


class MRUEvictionPolicy:
    """Evict the most recently released candidate first."""

    def __init__(self) -> None:
        self.inactive: list[str] = []

    def mark_active(self, key: str) -> None:
        if key in self.inactive:
            self.inactive.remove(key)

    def mark_inactive(self, key: str) -> None:
        self.mark_active(key)
        self.inactive.append(key)

    def discard(self, key: str) -> None:
        self.mark_active(key)

    def choose_victims(self, context: EvictionContext) -> tuple[str, ...]:
        sizes = {candidate.key: candidate.cache_bytes for candidate in context.candidates}
        selected: list[str] = []
        freed = 0
        for key in reversed(self.inactive):
            if key not in sizes:
                continue
            selected.append(key)
            freed += sizes[key]
            if freed >= context.bytes_to_free:
                break
        return tuple(selected)


@pytest.fixture(autouse=True)
def _reset_instances() -> None:
    FakeStore.instances.clear()


class TestConstruction:
    def test_negative_budget_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_cache_bytes"):
            ResourceCache(-1)

    def test_zero_budget_allowed(self) -> None:
        cache = ResourceCache(0)
        assert cache.max_cache_bytes == 0
        assert cache.used_cache_bytes == 0
        assert cache.available_cache_bytes == 0


class TestRegistration:
    def test_register_is_lazy(self) -> None:
        cache = ResourceCache(100)
        cache.register(_spec("a", 50))
        assert FakeStore.instances == []
        assert not cache.info("a").cached

    def test_duplicate_registration_rejected(self) -> None:
        cache = ResourceCache(100)
        cache.register(_spec("a", 50))
        with pytest.raises(DuplicateResourceKeyError):
            cache.register(_spec("a", 50))

    def test_replace_releases_cached_store(self) -> None:
        cache = ResourceCache(100)
        with cache.lease(_spec("a", 50)):
            pass
        cache.register(_spec("a", 25), replace=True)
        assert cache.used_cache_bytes == 0
        assert not cache.info("a").cached

    def test_replace_rejected_while_leased(self) -> None:
        cache = ResourceCache(100)
        with cache.lease(_spec("a", 50)):
            with pytest.raises(ResourceLeasedError, match="leased"):
                cache.register(_spec("a", 25), replace=True)

    def test_negative_estimate_rejected(self) -> None:
        cache = ResourceCache(100)
        with pytest.raises(ValueError, match="estimated_cache_bytes"):
            cache.register(
                FakeSpec(
                    key="a",
                    estimated_cache_bytes=-1,
                    factory=_factory(0),
                )
            )

    def test_unregister_unknown_is_noop(self) -> None:
        ResourceCache(100).unregister("missing")

    def test_unregister_rejected_while_leased(self) -> None:
        cache = ResourceCache(100)
        with cache.lease(_spec("a", 50)):
            with pytest.raises(ResourceLeasedError, match="leased"):
                cache.unregister("a")

    def test_unregister_can_require_explicit_eviction(self) -> None:
        cache = ResourceCache(100)
        with cache.lease(_spec("a", 50)):
            pass
        with pytest.raises(ResourceCachedError, match="built store"):
            cache.unregister("a", evict=False)
        cache.unregister("a")
        assert not _is_registered(cache, "a")


class TestLease:
    def test_accepts_structural_resource_spec_without_inheritance(self) -> None:
        @dataclass(frozen=True)
        class IndependentSpec:
            key: str
            estimated_cache_bytes: int
            factory: Callable[[], FakeStore]

            def build_store(self) -> ResourceStore:
                return self.factory()

            def value(self, store: ResourceStore) -> FakeStore:
                assert isinstance(store, FakeStore)
                return store

        cache = ResourceCache(100)
        spec = IndependentSpec("independent", 50, _factory(50))
        with cache.lease(spec) as store:
            assert isinstance(store, FakeStore)
            assert cache.info("independent").lease_count == 1

    def test_spec_auto_registers_and_factory_runs_once(self) -> None:
        cache = ResourceCache(100)
        spec = _spec("a", 50)
        with cache.lease(spec) as first:
            assert isinstance(first, FakeStore)
            assert cache.info("a").lease_count == 1
        with cache.lease("a") as second:
            assert second is first
        assert len(FakeStore.instances) == 1
        assert cache.info("a").lease_count == 0

    def test_unknown_key_rejected(self) -> None:
        cache = ResourceCache(100)
        with pytest.raises(ResourceNotRegisteredError, match="lease"):
            with cache.lease("missing"):
                pass

    def test_nested_same_key_counts_independent_leases(self) -> None:
        cache = ResourceCache(100)
        with cache.lease(_spec("a", 50)) as first:
            with cache.lease("a") as second:
                assert second is first
                assert cache.info("a").lease_count == 2
            assert cache.info("a").lease_count == 1
        assert cache.info("a").lease_count == 0

    def test_value_failure_releases_lease(self) -> None:
        class RaisingSpec(FakeSpec):
            def value(self, store: ResourceStore) -> FakeStore:
                raise RuntimeError("value boom")

        cache = ResourceCache(100)
        spec = RaisingSpec(
            key="a",
            estimated_cache_bytes=50,
            factory=_factory(50),
        )
        with pytest.raises(RuntimeError, match="value boom"):
            with cache.lease(spec):
                pass
        assert cache.info("a").lease_count == 0
        assert cache.info("a").cached

    def test_evict_and_clear_rejected_while_leased(self) -> None:
        cache = ResourceCache(100)
        with cache.lease(_spec("a", 50)):
            with pytest.raises(ResourceLeasedError, match="leased"):
                cache.evict("a")
            with pytest.raises(ResourceLeasedError, match="leased"):
                cache.clear()


class TestLeaseMany:
    def test_leases_in_order_and_releases_in_reverse(self) -> None:
        events: list[str] = []

        class RecordingPolicy(MRUEvictionPolicy):
            def mark_active(self, key: str) -> None:
                events.append(f"active:{key}")
                super().mark_active(key)

            def mark_inactive(self, key: str) -> None:
                events.append(f"inactive:{key}")
                super().mark_inactive(key)

        cache = ResourceCache(100, eviction_policy=RecordingPolicy())
        with cache.lease_many([_spec("a", 40), _spec("b", 40)]) as stores:
            assert len(stores) == 2
            assert cache.info("a").lease_count == 1
            assert cache.info("b").lease_count == 1

        release_events = [event for event in events if event.startswith("inactive:")]
        assert release_events[-2:] == ["inactive:b", "inactive:a"]

    def test_earlier_lease_is_protected_during_later_admission(self) -> None:
        cache = ResourceCache(100)
        with pytest.raises(ResourceTooLargeError):
            with cache.lease_many([_spec("dependency", 40), _spec("owner", 80)]):
                pass
        assert _is_cached(cache, "dependency")
        assert cache.info("dependency").lease_count == 0
        assert not _is_cached(cache, "owner")

    def test_partial_failure_releases_earlier_leases(self) -> None:
        cache = ResourceCache(100)
        bad = _spec("bad", 40, raises=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            with cache.lease_many([_spec("a", 40), bad]):
                pass
        assert cache.info("a").lease_count == 0


class TestEviction:
    def test_lru_evicts_oldest_released_store(self) -> None:
        cache = ResourceCache(100)
        for key in ("a", "b"):
            with cache.lease(_spec(key, 50)):
                pass
        with cache.lease(_spec("c", 50)):
            pass
        assert not _is_cached(cache, "a")
        assert _is_cached(cache, "b")
        assert _is_cached(cache, "c")

    def test_recent_reuse_refreshes_lru(self) -> None:
        cache = ResourceCache(100)
        for key in ("a", "b"):
            with cache.lease(_spec(key, 50)):
                pass
        with cache.lease("a"):
            pass
        with cache.lease(_spec("c", 50)):
            pass
        assert _is_cached(cache, "a")
        assert not _is_cached(cache, "b")

    def test_custom_policy_controls_victim(self) -> None:
        cache = ResourceCache(100, eviction_policy=MRUEvictionPolicy())
        for key in ("a", "b"):
            with cache.lease(_spec(key, 50)):
                pass
        with cache.lease(_spec("c", 50)):
            pass
        assert _is_cached(cache, "a")
        assert not _is_cached(cache, "b")

    @pytest.mark.parametrize(
        "victims",
        [("unknown",), ("a", "a")],
    )
    def test_invalid_policy_victims_rejected(self, victims: tuple[str, ...]) -> None:
        class InvalidPolicy(MRUEvictionPolicy):
            def choose_victims(self, context: EvictionContext) -> tuple[str, ...]:
                return victims

        cache = ResourceCache(50, eviction_policy=InvalidPolicy())
        with cache.lease(_spec("a", 50)):
            pass
        with pytest.raises(EvictionPolicyError, match="invalid"):
            with cache.lease(_spec("b", 50)):
                pass

    def test_insufficient_policy_victims_rejected(self) -> None:
        class EmptyPolicy(MRUEvictionPolicy):
            def choose_victims(self, context: EvictionContext) -> tuple[str, ...]:
                return ()

        cache = ResourceCache(50, eviction_policy=EmptyPolicy())
        with cache.lease(_spec("a", 50)):
            pass
        with pytest.raises(EvictionPolicyError, match="insufficient"):
            with cache.lease(_spec("b", 50)):
                pass

    def test_resource_larger_than_budget_rejected(self) -> None:
        cache = ResourceCache(50)
        with pytest.raises(ResourceTooLargeError):
            with cache.lease(_spec("a", 51)):
                pass

    def test_leased_store_blocks_eviction(self) -> None:
        cache = ResourceCache(100)
        with cache.lease(_spec("a", 75)):
            with pytest.raises(ResourceTooLargeError):
                with cache.lease(_spec("b", 50)):
                    pass


class TestFailuresAndAccounting:
    def test_factory_failure_preserves_registration(self) -> None:
        cache = ResourceCache(100)
        spec = _spec("bad", 50, raises=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            with cache.lease(spec):
                pass
        assert _is_registered(cache, "bad")
        assert not _is_cached(cache, "bad")
        assert cache.used_cache_bytes == 0

    def test_factory_failure_after_pre_eviction_keeps_eviction(self) -> None:
        cache = ResourceCache(100)
        with cache.lease(_spec("warm", 100)):
            pass
        bad = _spec("bad", 100, raises=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            with cache.lease(bad):
                pass
        assert not _is_cached(cache, "warm")
        assert not _is_cached(cache, "bad")

    def test_negative_actual_size_rejected(self) -> None:
        cache = ResourceCache(100)
        with pytest.raises(CacheError, match="cache_bytes"):
            with cache.lease(_spec("bad", -1, estimated_cache_bytes=10)):
                pass
        assert cache.used_cache_bytes == 0

    def test_actual_smaller_than_estimate_uses_actual(self) -> None:
        cache = ResourceCache(100)
        with cache.lease(_spec("a", 25, estimated_cache_bytes=75)):
            pass
        assert cache.used_cache_bytes == 25

    def test_actual_larger_than_estimate_evicts_more(self) -> None:
        cache = ResourceCache(100)
        with cache.lease(_spec("filler", 50)):
            pass
        with cache.lease(_spec("big", 75, estimated_cache_bytes=25)):
            pass
        assert not _is_cached(cache, "filler")
        assert cache.used_cache_bytes == 75

    def test_actual_larger_than_budget_is_rejected(self) -> None:
        cache = ResourceCache(100)
        with pytest.raises(ResourceTooLargeError):
            with cache.lease(_spec("big", 125, estimated_cache_bytes=25)):
                pass
        assert not _is_cached(cache, "big")
        assert cache.used_cache_bytes == 0


class TestThreadSafety:
    def test_concurrent_cache_miss_builds_once(self) -> None:
        cache = ResourceCache(100)
        spec = _spec("a", 50)
        barrier = threading.Barrier(8)
        errors: queue.Queue[BaseException] = queue.Queue()
        stores: queue.Queue[FakeStore] = queue.Queue()

        def worker() -> None:
            try:
                barrier.wait(timeout=2)
                with cache.lease(spec) as store:
                    stores.put(store)
            except BaseException as exc:
                errors.put(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)

        assert all(not thread.is_alive() for thread in threads)
        assert errors.empty()
        leased = [stores.get_nowait() for _ in range(stores.qsize())]
        assert len(leased) == 8
        assert len({id(store) for store in leased}) == 1
        assert len(FakeStore.instances) == 1
        assert cache.info("a").lease_count == 0


class TestObservabilityAndRelease:
    def test_info_reports_unbuilt_and_built_entries(self) -> None:
        cache = ResourceCache(100)
        cache.register(_spec("a", 50))
        info = cache.info("a")
        assert not info.cached
        assert info.cache_bytes is None
        assert info.lease_count == 0

        with cache.lease("a"):
            info = cache.info("a")
            assert info.cached
            assert info.cache_bytes == 50
            assert info.lease_count == 1

    def test_unknown_info_rejected(self) -> None:
        with pytest.raises(ResourceNotRegisteredError):
            ResourceCache(100).info("missing")

    def test_host_cache_callback_runs_after_positive_size_eviction(self) -> None:
        calls = 0

        def callback() -> None:
            nonlocal calls
            calls += 1

        cache = ResourceCache(100, empty_host_cache=callback)
        with cache.lease(_spec("a", 50)):
            pass
        assert calls == 0
        cache.evict("a")
        assert calls == 1

    def test_host_cache_callback_failure_is_ignored(self) -> None:
        def callback() -> None:
            raise RuntimeError("flush failed")

        cache = ResourceCache(100, empty_host_cache=callback)
        with cache.lease(_spec("a", 50)):
            pass
        cache.evict("a")


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab = {"hello": 0, "world": 1}


class TestObjectSpec:
    def test_factory_runs_once_and_leases_share_value(self) -> None:
        builds = 0

        def factory() -> FakeTokenizer:
            nonlocal builds
            builds += 1
            return FakeTokenizer()

        cache = ResourceCache(100)
        spec = ObjectSpec(key="tok", factory=factory)
        with cache.lease(spec) as first, cache.lease("tok") as second:
            assert second is first
            assert cache.info("tok").lease_count == 2
        assert builds == 1

    def test_zero_byte_default_does_not_consume_budget(self) -> None:
        cache = ResourceCache(0)
        with cache.lease(ObjectSpec(key="tok", factory=FakeTokenizer)):
            pass
        assert cache.used_cache_bytes == 0
        assert _is_cached(cache, "tok")

    def test_positive_estimate_counts_against_budget(self) -> None:
        cache = ResourceCache(100)
        with cache.lease(
            ObjectSpec(
                key="tok",
                factory=FakeTokenizer,
                estimated_cache_bytes=75,
            )
        ):
            pass
        with cache.lease(_spec("a", 50)):
            pass
        assert not _is_cached(cache, "tok")

    def test_store_wrapper_satisfies_resource_store(self) -> None:
        spec = ObjectSpec(key="tok", factory=FakeTokenizer)
        store = spec.build_store()
        assert isinstance(store, ResourceStore)
        assert isinstance(spec.value(store), FakeTokenizer)
