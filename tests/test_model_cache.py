"""Tests for ``torch_offload.model_cache.ModelCache``.

Uses a fake :class:`ModelStrategy` (``FakeStrategy``) to exercise the
cache without needing actual GPU memory or pinning. The fake records
every lifecycle call so tests can assert ordering and counts.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from torch import nn

from torch_offload import (
    ActivationError,
    DuplicateModelKeyError,
    ModelCache,
    ModelInUseError,
    ModelNotRegisteredError,
    ModelSpec,
    ModelTooLargeError,
)
from torch_offload.protocols import ModelStrategy


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

    def activate(self) -> None:
        self.events.append("activate")
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


# ---------------------------------------------------------------------------
# Hit / miss / build
# ---------------------------------------------------------------------------


class TestHitMiss:
    def test_first_use_builds_via_factory(self) -> None:
        cache = ModelCache(1000)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        s = cache.snapshot()
        assert s.stats.builds == 1
        assert s.stats.hits == 0
        assert s.stats.misses == 1
        assert s.used_cache_bytes == 100

    def test_second_use_hits_cache(self) -> None:
        cache = ModelCache(1000)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        with cache.use("a"):
            pass
        s = cache.snapshot()
        assert s.stats.builds == 1
        assert s.stats.hits == 1
        assert s.stats.misses == 1

    def test_use_with_spec_auto_registers(self) -> None:
        cache = ModelCache(1000)
        with cache.use(_spec("a", 100)):
            pass
        assert "a" in cache.snapshot().registered_keys

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
        first = FakeStrategy.instances[0]
        cache.register(_spec("a", 200), replace=True)
        # The freshly-replaced entry has not been built yet.
        assert cache.snapshot().used_cache_bytes == 0
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
        assert "a" not in cache.snapshot().registered_keys

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
        snap = cache.snapshot()
        assert snap.stats.evictions >= 1
        assert "a" not in snap.cached_keys_lru_to_mru

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
        snap = cache.snapshot()
        # Need to free at least 250-(300-300)=250, evicting 3*100=300.
        assert snap.stats.bytes_evicted >= 250

    def test_too_large_raises(self) -> None:
        cache = ModelCache(100)
        with pytest.raises(ModelTooLargeError) as excinfo:
            with cache.use(_spec("oversized", 200)):
                pass
        err = excinfo.value
        assert err.required == 200
        assert err.limit == 100

    def test_too_large_due_to_active_blockers_lists_them(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            # "a" is active and uses 100/200; "a" cannot be evicted.
            # Trying to admit a 150-byte "b" needs 50 more bytes than
            # currently free, but the only entry that could be evicted
            # is the active one. Should raise with "a" as the blocker.
            with pytest.raises(ModelTooLargeError) as excinfo:
                with cache.use(_spec("b", 150)):
                    pass
            assert "a" in excinfo.value.active_refcounts

    def test_active_entries_are_not_evicted(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        cache.register(_spec("b", 100))
        with cache.use("a"):
            with cache.use("b"):
                pass
        # "a" was active for the duration; only "b" became evictable.
        # Adding a third 100-byte while "a" still inactive should evict
        # whatever's at LRU.
        cache.register(_spec("c", 100))
        with cache.use("c"):
            pass
        # All three eventually built; "a" should still be one of the cached keys.
        snap = cache.snapshot()
        assert snap.stats.evictions >= 1

    def test_manual_evict(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        cache.evict("a")
        assert cache.snapshot().used_cache_bytes == 0

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
        assert cache.snapshot().used_cache_bytes == 0
        # Registrations preserved.
        assert set(cache.snapshot().registered_keys) == {"a", "b", "c"}

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
                snap = cache.snapshot()
                assert dict(snap.active_refcounts) == {"enc": 1, "dec": 1}

    def test_deactivate_returns_to_lru_at_mru(self) -> None:
        cache = ModelCache(300)
        for k in ["a", "b", "c"]:
            cache.register(_spec(k, 100))
            with cache.use(k):
                pass
        # After all uses, LRU order is a, b, c (oldest first).
        assert cache.snapshot().cached_keys_lru_to_mru == ("a", "b", "c")
        # Touching "a" makes it MRU.
        with cache.use("a"):
            pass
        assert cache.snapshot().cached_keys_lru_to_mru == ("b", "c", "a")


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
        snap = cache.snapshot()
        assert snap.used_cache_bytes == 0
        assert snap.stats.factory_errors == 1
        # Registration persisted (but no built strategy).
        assert "bad" in snap.registered_keys

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
        snap = cache.snapshot()
        assert snap.used_cache_bytes == 0
        assert snap.stats.activation_errors == 1
        # Strategy was constructed and then released.

    def test_activation_failure_on_cached_entry_discards_it(self) -> None:
        # Activation failure on a previously-cached entry is treated
        # as poisoned (same as freshly-built). Strategies like
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
        snap = cache.snapshot()
        assert "a" not in snap.cached_keys_lru_to_mru
        assert snap.used_cache_bytes == 0
        # The poisoned strategy reference was dropped.
        # Registration persisted for retry — but next acquire rebuilds.
        assert "a" in snap.registered_keys

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
        assert cache.snapshot().used_cache_bytes == 100
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
        snap = cache.snapshot()
        # warm was evicted to make room — that eviction is committed.
        assert "warm" not in snap.cached_keys_lru_to_mru
        assert snap.used_cache_bytes == 0
        # bad's registration persists for retry; warm's also.
        assert "warm" in snap.registered_keys
        assert "bad" in snap.registered_keys
        assert snap.stats.factory_errors == 1

# ---------------------------------------------------------------------------
# Deactivate failure
# ---------------------------------------------------------------------------


class TestDeactivateFailure:
    def test_deactivate_failure_discards_entry(self) -> None:
        # A strategy whose deactivate() raises is treated as poisoned:
        # the entry is removed from the cache and
        # the original deactivate exception propagates.
        class PoisonStrategy(FakeStrategy):
            def deactivate(self) -> None:
                self.events.append("deactivate")
                raise RuntimeError("deactivate boom")

        def factory() -> PoisonStrategy:
            s = PoisonStrategy(100)
            FakeStrategy.instances.append(s)
            return s

        cache = ModelCache(200)
        cache.register(ModelSpec(key="a", estimated_cache_bytes=100, factory=factory))
        with pytest.raises(RuntimeError, match="deactivate boom"):
            with cache.use("a"):
                pass
        snap = cache.snapshot()
        # Entry removed from cache, bytes reclaimed in accounting,
        # registration preserved for retry.
        assert "a" not in snap.cached_keys_lru_to_mru
        assert snap.used_cache_bytes == 0
        assert "a" in snap.registered_keys
        # The poisoned strategy reference was dropped.


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
        snap = cache.snapshot()
        assert snap.used_cache_bytes == 0
        # Strategy was constructed and then released.


# ---------------------------------------------------------------------------
# Snapshot is immutable
# ---------------------------------------------------------------------------


class TestSnapshotImmutability:
    def test_stats_in_snapshot_do_not_change_after_capture(self) -> None:
        cache = ModelCache(200)
        cache.register(_spec("a", 100))
        with cache.use("a"):
            pass
        snap_before = cache.snapshot()
        captured_hits = snap_before.stats.hits
        captured_builds = snap_before.stats.builds
        # Trigger more activity.
        with cache.use("a"):
            pass
        with cache.use("a"):
            pass
        # Captured snapshot's stats must not have changed.
        assert snap_before.stats.hits == captured_hits
        assert snap_before.stats.builds == captured_builds
        # But a new snapshot reflects current activity.
        snap_after = cache.snapshot()
        assert snap_after.stats.hits > captured_hits


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
        assert cache.snapshot().used_cache_bytes == 50

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
        snap = cache.snapshot()
        assert snap.used_cache_bytes == 250
        assert "filler" not in snap.cached_keys_lru_to_mru

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
        assert cache.snapshot().used_cache_bytes == 0


# ---------------------------------------------------------------------------
# Snapshot / info
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

    def test_peak_cache_bytes_tracked(self) -> None:
        cache = ModelCache(500)
        for k, b in [("a", 100), ("b", 200), ("c", 150)]:
            cache.register(_spec(k, b))
            with cache.use(k):
                pass
        peak = cache.snapshot().stats.peak_cache_bytes
        assert peak == 100 + 200 + 150


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
# ModelStrategy structural check
# ---------------------------------------------------------------------------


class TestStrategyConformance:
    def test_fake_satisfies_protocol(self) -> None:
        s = FakeStrategy(100)
        assert isinstance(s, ModelStrategy)
