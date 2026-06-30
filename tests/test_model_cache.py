"""Tests for ``torch_offload.model_cache.ModelCache``.

Uses a fake :class:`ResourceBinding` (``FakeBinding``) to exercise the
cache without needing actual GPU memory or pinning. The fake records
every lifecycle call so tests can assert ordering and counts.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import cast

import pytest
import torch
from torch import nn

from torch_offload import (
    DuplicateModelKeyError,
    EvictionContext,
    EvictionPolicy,
    EvictionPolicyError,
    ModelCache,
    ModelInUseError,
    ModelNotRegisteredError,
    ModelTooLargeError,
    ObjectSpec,
    ResourceSpec,
)
from torch_offload.protocols import ResourceBinding, ResourceStore

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Fake store/binding
# ---------------------------------------------------------------------------


class FakeStore:
    """In-memory fake satisfying the :class:`ResourceStore` protocol."""

    instances: list["FakeStore"] = []

    def __init__(
        self,
        cache_bytes: int,
        *,
        activate_raises: BaseException | None = None,
        bind_raises: BaseException | None = None,
    ) -> None:
        self._cache_bytes = cache_bytes
        self._activate_raises = activate_raises
        self._bind_raises = bind_raises
        FakeStore.instances.append(self)

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes

    def bind(self) -> "FakeBinding":
        if self._bind_raises is not None:
            raise self._bind_raises
        return FakeBinding(self, activate_raises=self._activate_raises)


class FakeBinding:
    """In-memory fake satisfying the :class:`ResourceBinding` protocol.

    Records every lifecycle call. Optionally raises on activate via
    an injected exception for failure-mode tests.
    """

    instances: list["FakeBinding"] = []

    def __init__(
        self,
        store: FakeStore,
        *,
        activate_raises: BaseException | None = None,
    ) -> None:
        self.store = store
        self._activate_raises = activate_raises
        self._active = False
        self.events: list[str] = []
        self.activate_devices: list[torch.device | None] = []
        self.module = nn.Identity()
        FakeBinding.instances.append(self)

    @property
    def model(self) -> nn.Module:
        return self.module

    @property
    def value(self) -> nn.Module:
        return self.module

    def activate(self, device: torch.device | str | None = None, **kwargs: object) -> None:
        del kwargs
        self.events.append("activate")
        self.activate_devices.append(torch.device(device) if device is not None else None)
        if self._activate_raises is not None:
            raise self._activate_raises
        if self._active:
            raise RuntimeError("FakeBinding already active")
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
    bind_raises: BaseException | None = None,
    factory_raises: BaseException | None = None,
) -> Callable[[], FakeStore]:
    def factory() -> FakeStore:
        if factory_raises is not None:
            raise factory_raises
        return FakeStore(
            bytes_,
            activate_raises=activate_raises,
            bind_raises=bind_raises,
        )

    return factory


def _bind_fake_store(store: ResourceStore) -> ResourceBinding[nn.Module]:
    return cast(FakeStore, store).bind()


def _spec(key: str, bytes_: int, **kwargs) -> ResourceSpec[nn.Module]:
    return ResourceSpec(
        key=key,
        estimated_cache_bytes=bytes_,
        store_factory=_make_factory(bytes_, **kwargs),
        bind=_bind_fake_store,
    )


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


@pytest.fixture(autouse=True)
def _clear_instances():
    FakeStore.instances.clear()
    FakeBinding.instances.clear()
    yield
    FakeStore.instances.clear()
    FakeBinding.instances.clear()


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
        assert len(FakeStore.instances) == 1
        assert len(FakeBinding.instances) == 1
        assert cache.info("a").cached
        assert cache.used_cache_bytes == 100
        assert cache.available_cache_bytes == 900

    def test_second_use_hits_cache(self) -> None:
        cache = ModelCache(1000)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        first = FakeBinding.instances[0]
        with cache.use("a"):
            pass
        assert len(FakeStore.instances) == 1
        assert len(FakeBinding.instances) == 2
        assert first.events == ["activate", "deactivate"]
        assert FakeBinding.instances[1].events == ["activate", "deactivate"]

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
# Active binding semantics
# ---------------------------------------------------------------------------


class TestActiveSet:
    def test_nested_same_key_creates_independent_bindings(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        with cache.use("a") as m1:
            with cache.use("a") as m2:
                assert m1 is not m2
                assert cache.info("a").active_count == 2
        assert len(FakeStore.instances) == 1
        assert len(FakeBinding.instances) == 2
        assert all(
            binding.events == ["activate", "deactivate"]
            for binding in FakeBinding.instances
        )

    def test_release_removes_exact_binding_when_equal(self) -> None:
        class EqualBinding(FakeBinding):
            def __eq__(self, other: object) -> bool:
                return isinstance(other, EqualBinding)

            __hash__ = object.__hash__

        class EqualBindingStore(FakeStore):
            def bind(self) -> EqualBinding:
                return EqualBinding(self)

        cache = ModelCache(200)
        cache.register(
            ResourceSpec(
                key="a",
                estimated_cache_bytes=100,
                store_factory=lambda: EqualBindingStore(100),
                bind=_bind_fake_store,
            )
        )
        with cache.use("a"):
            with cache.use("a"):
                pass
            assert cache.info("a").active_count == 1
            assert FakeBinding.instances[0].events == ["activate"]
            assert FakeBinding.instances[1].events == ["activate", "deactivate"]
        assert FakeBinding.instances[0].events == ["activate", "deactivate"]

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
# Device activation + thread-safe cache bindings
# ---------------------------------------------------------------------------


class TestDeviceActivationAndConcurrency:
    def test_different_cuda_devices_can_be_active_together(self) -> None:
        cache = ModelCache(300)
        cache.register(_spec("a", 100))
        cache.register(_spec("b", 100))

        with cache.use("a", device="cuda:0"):
            with cache.use("b", device="cuda:1"):
                assert _active_refcounts(cache, "a", "b") == {"a": 1, "b": 1}

        assert FakeBinding.instances[0].activate_devices == [torch.device("cuda:0")]
        assert FakeBinding.instances[1].activate_devices == [torch.device("cuda:1")]

    def test_different_keys_can_share_same_cuda_device(self) -> None:
        cache = ModelCache(300)
        cache.register(_spec("a", 100))
        cache.register(_spec("b", 100))

        with cache.use("a", device="cuda:0"):
            with cache.use("b", device="cuda:0"):
                assert _active_refcounts(cache, "a", "b") == {"a": 1, "b": 1}

        assert FakeBinding.instances[0].activate_devices == [torch.device("cuda:0")]
        assert FakeBinding.instances[1].activate_devices == [torch.device("cuda:0")]

    def test_different_key_can_reactivate_on_same_cuda_device_after_release(self) -> None:
        cache = ModelCache(300)
        cache.register(_spec("a", 100))
        cache.register(_spec("b", 100))

        with cache.use("a", device="cuda:0"):
            pass
        with cache.use("b", device="cuda:0"):
            pass

        assert FakeBinding.instances[0].activate_devices == [torch.device("cuda:0")]
        assert FakeBinding.instances[1].activate_devices == [torch.device("cuda:0")]

    def test_activation_failure_does_not_block_same_device_for_other_key(self) -> None:
        cache = ModelCache(300)
        cache.register(_spec("a", 100, activate_raises=RuntimeError("gpu boom")))
        cache.register(_spec("b", 100))

        with pytest.raises(RuntimeError, match="gpu boom"):
            with cache.use("a", device="cuda:0"):
                pass

        with cache.use("b", device="cuda:0"):
            pass

        assert FakeBinding.instances[1].activate_devices == [torch.device("cuda:0")]

    def test_different_keys_on_same_cuda_device_can_overlap_across_threads(self) -> None:
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
            threading.Thread(target=worker, args=("b", "cuda:0")),
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

        s = FakeBinding.instances[0]
        assert s.activate_devices == [torch.device("cpu")]

    def test_nested_same_device_creates_two_bindings(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))

        with cache.use("a", device="cpu"):
            with cache.use("a", device=torch.device("cpu")):
                pass

        assert [binding.activate_devices for binding in FakeBinding.instances] == [
            [torch.device("cpu")],
            [torch.device("cpu")],
        ]

    def test_nested_indexed_cpu_normalizes_device(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))

        with cache.use("a", device="cpu"):
            with cache.use("a", device="cpu:0"):
                pass

        assert [binding.activate_devices for binding in FakeBinding.instances] == [
            [torch.device("cpu")],
            [torch.device("cpu")],
        ]

    def test_nested_omitted_device_gets_independent_binding(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))

        with cache.use("a", device="cpu"):
            with cache.use("a"):
                pass

        assert [binding.activate_devices for binding in FakeBinding.instances] == [
            [torch.device("cpu")],
            [None],
        ]

    def test_nested_different_device_allowed_with_independent_binding(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))

        with cache.use("a", device="cpu"):
            with cache.use("a", device="meta"):
                pass

        assert [binding.activate_devices for binding in FakeBinding.instances] == [
            [torch.device("cpu")],
            [torch.device("meta")],
        ]

    def test_nested_device_after_unspecified_activation_allowed(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))

        with cache.use("a"):
            with cache.use("a", device="cpu"):
                pass

        assert [binding.activate_devices for binding in FakeBinding.instances] == [
            [None],
            [torch.device("cpu")],
        ]

    def test_inactive_entry_can_reactivate_on_different_device(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))

        with cache.use("a", device="cpu"):
            pass
        with cache.use("a", device="meta"):
            pass

        assert [binding.activate_devices for binding in FakeBinding.instances] == [
            [torch.device("cpu")],
            [torch.device("meta")],
        ]

    @CUDA
    def test_reentrant_bare_cuda_matches_current_indexed_cuda(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        expected = torch.device("cuda", torch.cuda.current_device())

        with cache.use("a", device="cuda"):
            with cache.use("a", device=expected):
                pass

        assert [binding.activate_devices for binding in FakeBinding.instances] == [
            [expected],
            [expected],
        ]


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailureModes:
    def test_factory_failure_leaves_cache_unchanged(self) -> None:
        cache = ModelCache(200)
        spec = ResourceSpec(
            key="bad",
            estimated_cache_bytes=100,
            store_factory=_make_factory(100, factory_raises=RuntimeError("boom")),
            bind=_bind_fake_store,
        )
        with pytest.raises(RuntimeError, match="boom"):
            with cache.use(spec):
                pass
        assert cache.used_cache_bytes == 0
        # Registration persisted (but no built store).
        assert _is_registered(cache, "bad")
        assert not _is_cached(cache, "bad")

    def test_activation_failure_on_fresh_build_keeps_store(self) -> None:
        # Activation failure discards only the failed binding. The
        # backing store remains cached for a retry.
        cache = ModelCache(200)
        spec = ResourceSpec(
            key="bad",
            estimated_cache_bytes=100,
            store_factory=_make_factory(100, activate_raises=RuntimeError("gpu boom")),
            bind=_bind_fake_store,
        )
        with pytest.raises(RuntimeError, match="gpu boom"):
            with cache.use(spec):
                pass
        assert cache.used_cache_bytes == 100
        assert _is_registered(cache, "bad")
        assert _is_cached(cache, "bad")
        assert cache.info("bad").active_count == 0

    def test_activation_failure_on_cached_entry_keeps_store(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        store = FakeStore.instances[0]
        store._activate_raises = RuntimeError("transient gpu")
        with pytest.raises(RuntimeError, match="transient gpu"):
            with cache.use("a"):
                pass
        assert _is_cached(cache, "a")
        assert cache.used_cache_bytes == 100
        assert cache.info("a").active_count == 0
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
        bad_spec = ResourceSpec(
            key="bad",
            estimated_cache_bytes=100,
            store_factory=_make_factory(100, factory_raises=RuntimeError("build boom")),
            bind=_bind_fake_store,
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
    def test_deactivate_failure_discards_binding_only(self) -> None:
        # A binding whose deactivate() raises is unrecoverable, but the
        # backing store remains cached.
        class FailingDeactivateBinding(FakeBinding):
            def deactivate(self) -> None:
                self.events.append("deactivate")
                raise RuntimeError("deactivate boom")

        class FailingDeactivateStore(FakeStore):
            def bind(self) -> FailingDeactivateBinding:
                return FailingDeactivateBinding(self)

        def factory() -> FailingDeactivateStore:
            return FailingDeactivateStore(100)

        cache = ModelCache(200)
        cache.register(
            ResourceSpec(
                key="a",
                estimated_cache_bytes=100,
                store_factory=factory,
                bind=_bind_fake_store,
            )
        )
        with pytest.raises(RuntimeError, match="deactivate boom"):
            with cache.use("a"):
                pass
        assert _is_cached(cache, "a")
        assert cache.used_cache_bytes == 100
        assert _is_registered(cache, "a")
        assert cache.info("a").active_count == 0


# ---------------------------------------------------------------------------
# cache_bytes validation
# ---------------------------------------------------------------------------


class TestCacheBytesValidation:
    def test_negative_actual_rejected(self) -> None:
        def factory():
            return FakeStore(-1)

        cache = ModelCache(200)
        spec = ResourceSpec(
            key="bad",
            estimated_cache_bytes=10,
            store_factory=factory,
            bind=_bind_fake_store,
        )
        with pytest.raises(Exception, match="cache_bytes"):
            with cache.use(spec):
                pass
        assert cache.used_cache_bytes == 0
        assert _is_registered(cache, "bad")
        assert not _is_cached(cache, "bad")
        # Store was constructed and then released.

    def test_bind_failure_keeps_store(self) -> None:
        cache = ModelCache(200)
        spec = _spec("bad", 100, bind_raises=RuntimeError("bind boom"))
        with pytest.raises(RuntimeError, match="bind boom"):
            with cache.use(spec):
                pass
        assert cache.used_cache_bytes == 100
        assert _is_cached(cache, "bad")
        assert cache.info("bad").active_count == 0

    def test_value_failure_after_activate_releases_binding(self) -> None:
        class ValueErrorBinding(FakeBinding):
            @property
            def value(self) -> nn.Module:
                raise RuntimeError("value boom")

        class ValueErrorStore(FakeStore):
            def bind(self) -> ValueErrorBinding:
                return ValueErrorBinding(self)

        cache = ModelCache(200)
        spec = ResourceSpec(
            key="bad",
            estimated_cache_bytes=100,
            store_factory=lambda: ValueErrorStore(100),
            bind=_bind_fake_store,
        )
        with pytest.raises(RuntimeError, match="value boom"):
            with cache.use(spec):
                pass
        assert _is_cached(cache, "bad")
        assert cache.info("bad").active_count == 0
        assert FakeBinding.instances[0].events == ["activate", "deactivate"]


# ---------------------------------------------------------------------------
# Actual-vs-estimate reconciliation
# ---------------------------------------------------------------------------


class TestActualVsEstimate:
    def test_actual_within_estimate_uses_estimate(self) -> None:
        # No reconciliation needed if actual <= estimate.
        cache = ModelCache(200)

        def factory():
            return FakeStore(50)  # smaller than estimate

        cache.register(
            ResourceSpec(
                key="a",
                estimated_cache_bytes=100,
                store_factory=factory,
                bind=_bind_fake_store,
            )
        )
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
            return FakeStore(250)  # actual is 250 vs estimated 150

        cache.register(
            ResourceSpec(
                key="big",
                estimated_cache_bytes=150,
                store_factory=big_factory,
                bind=_bind_fake_store,
            ),
        )
        with cache.use("big"):
            pass
        # "filler" must have been evicted to make room for the actual 250.
        assert cache.used_cache_bytes == 250
        assert not _is_cached(cache, "filler")

    def test_actual_overflow_rejects_and_releases(self) -> None:
        cache = ModelCache(100)

        def oversized():
            return FakeStore(200)  # estimate 50, actual 200

        spec = ResourceSpec(
            key="bad",
            estimated_cache_bytes=50,
            store_factory=oversized,
            bind=_bind_fake_store,
        )
        with pytest.raises(ModelTooLargeError):
            with cache.use(spec):
                pass
        # The constructed store reference was dropped.
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
# ResourceBinding structural check
# ---------------------------------------------------------------------------


class TestStrategyConformance:
    def test_fake_satisfies_protocol(self) -> None:
        store = FakeStore(100)
        s = FakeBinding(store)
        assert isinstance(store, ResourceStore)
        assert isinstance(s, ResourceBinding)


# ---------------------------------------------------------------------------
# ObjectSpec
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Stand-in for a non-module Python object (tokenizer, processor)."""

    def __init__(self) -> None:
        self.vocab = {"hello": 0, "world": 1}


class TestObjectSpec:
    def test_factory_runs_once_and_instance_is_shared(self) -> None:
        builds = {"n": 0}

        def factory() -> FakeTokenizer:
            builds["n"] += 1
            return FakeTokenizer()

        cache = ModelCache(1000)
        spec = ObjectSpec(key="tok", factory=factory)
        with cache.use(spec) as first:
            assert isinstance(first, FakeTokenizer)
        with cache.use("tok") as second:
            assert second is first
        assert builds["n"] == 1

    def test_concurrent_bindings_share_instance(self) -> None:
        cache = ModelCache(1000)
        spec = ObjectSpec(key="tok", factory=FakeTokenizer)
        with cache.use(spec) as outer, cache.use("tok") as inner:
            assert inner is outer
            assert cache.info("tok").active_count == 2

    def test_device_is_ignored(self) -> None:
        cache = ModelCache(1000)
        spec = ObjectSpec(key="tok", factory=FakeTokenizer)
        with cache.use(spec, device="cpu") as tok:
            assert isinstance(tok, FakeTokenizer)

    def test_zero_byte_default_does_not_consume_budget(self) -> None:
        cache = ModelCache(0)
        spec = ObjectSpec(key="tok", factory=FakeTokenizer)
        with cache.use(spec):
            pass
        assert cache.used_cache_bytes == 0
        assert _is_cached(cache, "tok")

    def test_zero_byte_entry_survives_lru_pressure(self) -> None:
        cache = ModelCache(200)
        cache.register(ObjectSpec(key="tok", factory=FakeTokenizer))
        cache.register(_spec("a", 100))
        cache.register(_spec("b", 150))
        with cache.use("tok"):
            pass
        with cache.use("a"):
            pass
        # Admitting "b" must evict; "tok" is the LRU-oldest inactive key
        # but frees nothing, so the policy must skip it and evict "a".
        with cache.use("b"):
            pass
        assert _is_cached(cache, "tok")
        assert not _is_cached(cache, "a")
        assert _is_cached(cache, "b")

    def test_positive_estimate_counts_against_budget(self) -> None:
        cache = ModelCache(200)
        cache.register(ObjectSpec(key="big", factory=FakeTokenizer, estimated_cache_bytes=150))
        cache.register(_spec("a", 100))
        with cache.use("big"):
            pass
        assert cache.used_cache_bytes == 150
        # Admitting "a" forces eviction of the sized object entry.
        with cache.use("a"):
            pass
        assert not _is_cached(cache, "big")

    def test_zero_byte_eviction_skips_host_cache_flush(self) -> None:
        calls = {"n": 0}
        cache = ModelCache(200, empty_host_cache=lambda: calls.__setitem__("n", calls["n"] + 1))
        cache.register(ObjectSpec(key="tok", factory=FakeTokenizer))
        with cache.use("tok"):
            pass
        cache.evict("tok")
        assert calls["n"] == 0

    def test_store_satisfies_protocols(self) -> None:
        spec = ObjectSpec(key="tok", factory=FakeTokenizer)
        store = spec.store_factory()
        assert isinstance(store, ResourceStore)
        binding = spec.bind(store)
        assert binding is store  # the store is its own binding
        assert isinstance(binding.value, FakeTokenizer)
        binding.activate()
        binding.deactivate()
