"""Tests for the block-streaming machinery in ``torch_offload``.

Covers ``ModelOffloader`` (the public composite),
``StreamedWeights`` (the per-block-list primitive),
``TrainableWeights`` (the trainable-param component),
and the cross-region tied-weight detector.

Most lifecycle tests run on CPU (the machinery is device-agnostic);
CUDA-only tests gate on availability.
"""

from __future__ import annotations

from concurrent.futures import Future

import pytest
import torch
from torch import nn

from torch_offload import (
    ModelOffloader,
    ModelStrategy,
    PinnedWeights,
    SlotOwnership,
    StreamedWeights,
    TrainableWeights,
)
from torch_offload.model_offloader import detect_streaming_region_ties
from torch_offload.slots import iter_buffer_slots
from torch_offload.streamed_weights import _BlockPinnedStore

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_block_model(num_blocks: int = 4, width: int = 8) -> nn.Module:
    """Tiny transformer-shaped model: nn.ModuleList of Linear blocks
    plus a non-block embed/head for non-block-resident testing.
    All params frozen."""

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(width, width, bias=False)
            self.transformer_blocks = nn.ModuleList(
                [nn.Linear(width, width, bias=False) for _ in range(num_blocks)]
            )
            self.head = nn.Linear(width, width, bias=False)

        def forward(self, x):
            x = self.embed(x)
            for block in self.transformer_blocks:
                x = block(x)
            return self.head(x)

    m = TinyModel()
    for p in m.parameters():
        p.requires_grad = False
    return m


# ---------------------------------------------------------------------------
# ModelStrategy conformance
# ---------------------------------------------------------------------------


class TestModelStrategyConformance:
    def test_isinstance_runtime_check(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m, torch.device("cpu"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            assert isinstance(strategy, ModelStrategy)
        finally:
            strategy.deactivate()

    def test_has_lifecycle_methods(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m, torch.device("cpu"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            assert callable(strategy.activate)
            assert callable(strategy.deactivate)
            assert isinstance(strategy.cache_bytes, int)
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Constructor pins; cache_bytes is final immediately
# ---------------------------------------------------------------------------


class TestConstructorPins:
    def test_constructor_pins_blocks(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m, torch.device("cpu"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            # Block weights are pinned via swapped slots.
            for block in m.transformer_blocks:
                assert block.weight.is_pinned()
            # Non-block (embed/head) also pinned via composed PinnedWeights.
            assert m.embed.weight.is_pinned()
            assert m.head.weight.is_pinned()
            assert strategy.cache_bytes > 0
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Lifecycle: constructor → activate → deactivate
# ---------------------------------------------------------------------------


class TestLifecycle:
    @CUDA
    def test_activate_returns_model(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.activate()
            assert strategy.model is m
        finally:
            strategy.deactivate()

    @CUDA
    def test_activate_brings_non_block_to_gpu(self) -> None:
        m = _make_block_model()
        target = torch.device("cuda")
        strategy = ModelOffloader(
            m, target, layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.activate()
            assert m.embed.weight.is_cuda
            assert m.head.weight.is_cuda
        finally:
            strategy.deactivate()

    @CUDA
    def test_deactivate_returns_non_block_to_pinned(self) -> None:
        m = _make_block_model()
        target = torch.device("cuda")
        strategy = ModelOffloader(
            m, target, layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.activate()
            assert m.embed.weight.is_cuda
            strategy.deactivate()
            assert m.embed.weight.device != target
            assert m.embed.weight.is_pinned()
            assert m.head.weight.is_pinned()
        finally:
            strategy.deactivate()

    @CUDA
    def test_reactivation_cycle(self) -> None:
        m = _make_block_model()
        target = torch.device("cuda")
        strategy = ModelOffloader(
            m, target, layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            cache_bytes = strategy.cache_bytes
            strategy.activate()
            strategy.deactivate()
            assert strategy.cache_bytes == cache_bytes
            strategy.activate()
            strategy.deactivate()
        finally:
            strategy.deactivate()

    def test_deactivate_when_not_active_is_noop(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m, torch.device("cpu"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.deactivate()  # no error, never activated
            strategy.deactivate()  # still no error
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Cleanup contract: deactivate restores CPU state, drop refs to free
# ---------------------------------------------------------------------------


class TestCleanup:
    @CUDA
    def test_deactivate_restores_cpu_state(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strategy.activate()
        strategy.deactivate()
        # Model is back in CPU/pinned state — usable, just without the
        # strategy's GPU streaming.
        for p in m.parameters():
            assert not p.is_cuda

    def test_deactivate_is_idempotent(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m, torch.device("cpu"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strategy.deactivate()
        strategy.deactivate()  # no error

    @CUDA
    def test_deactivate_consumes_teardown_stack(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strategy.activate()
        assert strategy._teardown_stack is not None
        strategy.deactivate()
        assert strategy._teardown_stack is None

    @CUDA
    def test_drop_strategy_without_deactivate_does_not_cycle(self) -> None:
        # Regression: StreamedWeights's forward-pre-hook closure used to
        # capture `self`, creating a refcount cycle:
        #     layer → _forward_pre_hooks → closure → streamer →
        #     _blocks → layer
        # Refcount-based GC couldn't break it; only the periodic cycle
        # collector could. We use weakref.ref(self) in the closure to
        # break the cycle so dropping the strategy frees it
        # immediately even if the user skipped deactivate().
        import gc
        import weakref

        m = _make_block_model()
        strategy = ModelOffloader(
            m, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strategy.activate()  # installs hooks; no deactivate
        # Reach the streamer (last component).
        streamer = strategy._components[-1]
        streamer_ref = weakref.ref(streamer)

        # Disable cycle collector BEFORE dropping refs to prove the
        # cleanup is purely refcount-based, not cycle-collector-based.
        gc.disable()
        try:
            del strategy, streamer
            assert streamer_ref() is None
        finally:
            gc.enable()

    @CUDA
    def test_orphaned_hooks_noop_after_strategy_dropped(self) -> None:
        # When the strategy is dropped without deactivate, the hooks
        # remain installed on the model but the weakref inside is
        # dead. The hook must no-op cleanly so the model still works
        # for forward. (Slow path: blocks may be on GPU or pinned-CPU
        # depending on eviction state at drop-time.)
        torch.manual_seed(0)
        m = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader(
            m, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strategy.activate()
        # Drop without deactivate.
        del strategy

        # Forward through the model — orphaned hooks should no-op.
        # (Some blocks resident on GPU, some pinned-CPU. PyTorch will
        # auto-promote inputs to match param device for each block.)
        x = torch.randn(2, 8, device="cpu")
        with torch.no_grad():
            # Should not raise. We don't assert correctness of the
            # output (params are in a mixed GPU/pinned-CPU state) —
            # just that the hooks don't crash.
            try:
                _ = m(x)
            except Exception as e:
                # If the cuda blocks fail because of input device
                # mismatch, that's expected. The hook itself must not
                # crash with AttributeError or NoneType errors.
                if "AttributeError" in repr(e) or "NoneType" in repr(e):
                    raise AssertionError(f"orphan hook crashed: {e!r}") from e


# ---------------------------------------------------------------------------
# Hook lifecycle (CUDA-only)
# ---------------------------------------------------------------------------


class TestHookLifecycle:
    @CUDA
    def test_hooks_installed_on_activate_removed_on_deactivate(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.activate()
            for block in m.transformer_blocks:
                assert len(block._forward_pre_hooks) > 0
            strategy.deactivate()
            for block in m.transformer_blocks:
                assert len(block._forward_pre_hooks) == 0
        finally:
            strategy.deactivate()

    @CUDA
    def test_hooks_removed_on_deactivate_drop(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strategy.activate()
        blocks = list(m.transformer_blocks)
        strategy.deactivate()
        for block in blocks:
            assert len(block._forward_pre_hooks) == 0


# ---------------------------------------------------------------------------
# Cyclic prefetch
# ---------------------------------------------------------------------------


class TestCyclicPrefetch:
    """Cyclic mode: iteration loops over the same block list (diffusion,
    multi-step decoders) detect end-of-iteration as wraparound rather
    than direction reversal, and prefetch indices wrap modulo N.
    """

    def _record_prefetches(
        self, streamer: StreamedWeights,
    ) -> tuple[list[int], object]:
        """Wrap streamer._submit_prefetch to record idx without disabling
        the actual prefetch (so on-GPU residency stays consistent)."""
        recorded: list[int] = []
        original = streamer._submit_prefetch

        def record(idx: int, max_on_gpu: int) -> None:
            recorded.append(idx)
            original(idx, max_on_gpu)

        streamer._submit_prefetch = record  # type: ignore[method-assign]
        return recorded, original

    @CUDA
    def test_cyclic_mode_wraps_prefetch_at_iteration_boundary(self) -> None:
        # Two iterations through 4 blocks with cyclic=True. Second
        # iteration's idx=0 hook must submit prefetches for 1 and 2
        # (forward direction inferred from wraparound), not -1 and -2.
        m = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader(
            m, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=1,
            prefetch_count=2, cyclic=True,
        )
        streamer: StreamedWeights = strategy._streamers[0]

        with strategy:
            recorded, _ = self._record_prefetches(streamer)
            x = torch.randn(2, 8, device="cuda")
            m(x)  # iteration 1
            torch.cuda.synchronize()
            recorded.clear()
            m(x)  # iteration 2
            torch.cuda.synchronize()

        # 4 blocks * 2 prefetches per hook = 8 entries.
        # Per-hook expected (cyclic, prefetch_count=2):
        #   idx=0 (last=3, |Δ|=3 > 2 → wrap-forward): 1, 2
        #   idx=1 (last=0, Δ=1 → forward):           2, 3
        #   idx=2 (last=1, Δ=1 → forward):           3, 0 (wrap)
        #   idx=3 (last=2, Δ=1 → forward):           0, 1 (wrap)
        assert recorded == [1, 2, 2, 3, 3, 0, 0, 1], recorded

    @CUDA
    def test_non_cyclic_mode_misfires_at_iteration_boundary(self) -> None:
        # Documented prior behavior preserved: with cyclic=False, the
        # second iteration's idx=0 hook detects backward direction
        # (because last_idx=N-1) and submits negative indices that
        # _submit_prefetch's bounds check drops. Asserting the misfire
        # so future cyclic-default changes are caught.
        m = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader(
            m, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=1,
            prefetch_count=2, cyclic=False,
        )
        streamer: StreamedWeights = strategy._streamers[0]

        with strategy:
            recorded, _ = self._record_prefetches(streamer)
            x = torch.randn(2, 8, device="cuda")
            m(x)  # iteration 1
            torch.cuda.synchronize()
            recorded.clear()
            m(x)  # iteration 2
            torch.cuda.synchronize()

        # idx=0 hook in iteration 2: last=3 → 0 < 3 → backward,
        # prefetch -1, -2 (no-op via bounds check downstream).
        assert recorded[0] == -1
        assert recorded[1] == -2

    @CUDA
    def test_cyclic_mode_continuous_backward_stays_backward(self) -> None:
        # Step-by-step reverse traversal (3, 2, 1, 0): each Δ=-1 is
        # below the wrap threshold, so direction inference yields
        # backward — not wrap-forward. Prefetch indices wrap modulo
        # N when the target falls below 0.
        m = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader(
            m, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=1,
            prefetch_count=1, cyclic=True,
        )
        streamer: StreamedWeights = strategy._streamers[0]

        with strategy:
            recorded, _ = self._record_prefetches(streamer)
            x = torch.randn(2, 8, device="cuda")
            for idx in (3, 2, 1, 0):
                streamer._blocks[idx](x)
            torch.cuda.synchronize()

        # Per-hook (cyclic, prefetch_count=1):
        #   idx=3 (last=-1 → forward init): prefetch 4 → wrap to 0
        #   idx=2 (last=3, Δ=-1 → backward): prefetch 1
        #   idx=1 (last=2, Δ=-1 → backward): prefetch 0
        #   idx=0 (last=1, Δ=-1 → backward): prefetch -1 → wrap to 3
        assert recorded == [0, 1, 0, 3], recorded

    @CUDA
    def test_cyclic_mode_two_iterations_match_eager_baseline(self) -> None:
        # Forward correctness: cyclic prefetch must not change which
        # block executes for which input. Two iterations through the
        # offloaded model must produce identical outputs to two
        # iterations of the same model on GPU without offloading.
        torch.manual_seed(42)
        m_eager = _make_block_model(num_blocks=4, width=8).cuda()
        x = torch.randn(2, 8, device="cuda")
        with torch.no_grad():
            expected_1 = m_eager(x)
            expected_2 = m_eager(expected_1)

        torch.manual_seed(42)
        m_off = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader(
            m_off, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=1,
            prefetch_count=2, cyclic=True,
        )
        try:
            with strategy:
                with torch.no_grad():
                    got_1 = m_off(x)
                    got_2 = m_off(got_1)
                torch.cuda.synchronize()
            torch.testing.assert_close(got_1, expected_1, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(got_2, expected_2, atol=1e-5, rtol=1e-5)
        finally:
            strategy.deactivate()

    @CUDA
    def test_cyclic_mode_small_n_threshold_corner(self) -> None:
        # num_blocks=3 puts wrap_threshold at 1, the smallest meaningful
        # threshold (any |Δ|>1 wraps). Locks in current behavior at
        # this corner: forward continuation uses Δ=1 (no wrap), and
        # iteration boundary 2→0 has |Δ|=2>1 (wraps to forward).
        m = _make_block_model(num_blocks=3, width=8)
        strategy = ModelOffloader(
            m, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=1,
            prefetch_count=1, cyclic=True,
        )
        streamer: StreamedWeights = strategy._streamers[0]

        with strategy:
            recorded, _ = self._record_prefetches(streamer)
            x = torch.randn(2, 8, device="cuda")
            m(x)  # iteration 1
            torch.cuda.synchronize()
            recorded.clear()
            m(x)  # iteration 2
            torch.cuda.synchronize()

        # 3 blocks * 1 prefetch each:
        #   idx=0 (last=2, |Δ|=2>1 → wrap-forward): prefetch 1
        #   idx=1 (last=0, Δ=1 → forward):          prefetch 2
        #   idx=2 (last=1, Δ=1 → forward):          prefetch 3 → wrap to 0
        assert recorded == [1, 2, 0], recorded


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


class TestForwardCorrectness:
    @CUDA
    def test_forward_matches_eager_baseline(self) -> None:
        torch.manual_seed(42)
        m = _make_block_model(num_blocks=4, width=8).cuda()
        x = torch.randn(2, 8, device="cuda")
        with torch.no_grad():
            expected = m(x)

        torch.manual_seed(42)
        m_off = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader(
            m_off, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.activate()
            with torch.no_grad():
                got = m_off(x)
            torch.cuda.synchronize()
            torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
        finally:
            strategy.deactivate()

    @CUDA
    def test_forward_after_deactivate_then_activate_cycle(self) -> None:
        torch.manual_seed(42)
        m = _make_block_model(num_blocks=4, width=8)
        strategy = ModelOffloader(
            m, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            x = torch.randn(2, 8, device="cuda")
            strategy.activate()
            with torch.no_grad():
                first = m(x)
            torch.cuda.synchronize()
            strategy.deactivate()

            strategy.activate()
            with torch.no_grad():
                second = m(x)
            torch.cuda.synchronize()
            torch.testing.assert_close(first, second)
            strategy.deactivate()
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_blocks_to_swap_must_be_lt_num_layers(self) -> None:
        m = _make_block_model(num_blocks=4)
        with pytest.raises(ValueError, match="blocks_to_swap"):
            ModelOffloader(
                m, torch.device("cpu"),
                layers_attr="transformer_blocks", blocks_to_swap=4,  # equal to num_blocks
            )

    def test_empty_layers_attr_raises(self) -> None:
        m = _make_block_model(num_blocks=4)
        with pytest.raises(ValueError, match="at least one path"):
            ModelOffloader(
                m, torch.device("cpu"),
                layers_attr=[], blocks_to_swap=2,
            )

    def test_layers_attr_resolving_to_non_modulelist_raises(self) -> None:
        m = _make_block_model(num_blocks=4)
        with pytest.raises(TypeError, match="nn.ModuleList"):
            ModelOffloader(
                m, torch.device("cpu"),
                layers_attr="embed",  # an nn.Linear, not a ModuleList
                blocks_to_swap=2,
            )


# ---------------------------------------------------------------------------
# ModelCache integration: cache_bytes is final at construction
# ---------------------------------------------------------------------------


class TestModelCacheIntegration:
    @CUDA
    def test_factory_returns_strategy_with_final_cache_bytes(self) -> None:
        from torch_offload import ModelCache, ModelSpec

        device = torch.device("cuda")

        def factory():
            m = _make_block_model(num_blocks=4, width=8)
            return ModelOffloader(
                m, device, layers_attr="transformer_blocks", blocks_to_swap=2,
            )

        cache = ModelCache(max_cache_bytes=10_000_000)
        spec = ModelSpec(key="xformer", estimated_cache_bytes=1024, factory=factory)

        with cache.use(spec) as model:
            assert isinstance(model, nn.Module)
            x = torch.randn(2, 8, device=device)
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()

        info = cache.info("xformer")
        assert info.cached
        assert info.cache_bytes is not None
        assert info.cache_bytes > 0
        assert info.active_count == 0

        # Second use is a cache hit — no rebuild.
        with cache.use("xformer"):
            pass
        snap = cache.snapshot()
        assert snap.stats.builds == 1
        assert snap.stats.hits == 1

        cache.clear()


# ---------------------------------------------------------------------------
# Activate failure → poison contract
# ---------------------------------------------------------------------------


class TestActivateFailurePoison:
    @CUDA
    def test_partial_activate_failure_rolls_back_other_components(self, monkeypatch) -> None:
        # If a streamer's activate raises, the composite's `with stack:`
        # rolls back the already-activated components (PinnedWeights,
        # TrainableWeights). _teardown_stack stays None because pop_all()
        # was never reached. Caller's responsibility to drop the
        # strategy reference for full cleanup.
        m = _make_block_model()
        strategy = ModelOffloader(
            m, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        streamer: StreamedWeights = strategy._components[-1]
        original_register_hooks = streamer._register_hooks

        def broken_register_hooks(*args, **kwargs):
            original_register_hooks(*args, **kwargs)
            raise RuntimeError("simulated activate failure")

        monkeypatch.setattr(streamer, "_register_hooks", broken_register_hooks)

        with pytest.raises(RuntimeError, match="simulated activate failure"):
            strategy.activate()

        assert strategy._teardown_stack is None

    def test_failing_components_own_deactivate_runs(self) -> None:
        # Regression: previously the composite registered each
        # component's deactivate AFTER calling its activate. If a
        # component's activate raised mid-way, only PRIOR siblings'
        # deactivates ran — the failing component's own cleanup was
        # skipped, leaking partial state. The fix registers deactivate
        # FIRST, so the failing component is also unwound.
        events: list[str] = []

        class _Recorder:
            def __init__(self, name: str, raise_on_activate: bool = False):
                self._name = name
                self._raise = raise_on_activate

            cache_bytes = 0

            def activate(self) -> None:
                events.append(f"activate:{self._name}")
                if self._raise:
                    # Simulate partial mutation before failing.
                    events.append(f"partial_state:{self._name}")
                    raise RuntimeError(f"{self._name} activate failed")

            def deactivate(self) -> None:
                events.append(f"deactivate:{self._name}")

        m = _make_block_model()
        strat = ModelOffloader(
            m, torch.device("cpu"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strat._components = [_Recorder("A"), _Recorder("B"), _Recorder("C", raise_on_activate=True)]

        with pytest.raises(RuntimeError, match="C activate failed"):
            strat.activate()

        # Activates ran in order until C raised.
        assert "activate:A" in events
        assert "activate:B" in events
        assert "activate:C" in events
        assert "partial_state:C" in events
        # Critical: C's own deactivate MUST run (the regression's bug
        # was that it didn't). Plus prior siblings' deactivates run in
        # reverse order via ExitStack.
        assert "deactivate:C" in events
        assert "deactivate:B" in events
        assert "deactivate:A" in events
        # Reverse order on unwind.
        deact_events = [e for e in events if e.startswith("deactivate:")]
        assert deact_events == ["deactivate:C", "deactivate:B", "deactivate:A"]


# ---------------------------------------------------------------------------
# Prefetch failure during deactivate
# ---------------------------------------------------------------------------


class TestPrefetchFailureOnDeactivate:
    @CUDA
    def test_prefetch_failure_propagates_after_cleanup(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strategy.activate()
        streamer: StreamedWeights = strategy._components[-1]
        # Inject a pre-failed Future so deactivate's drain loop hits it.
        bad_future: Future[None] = Future()
        bad_future.set_exception(RuntimeError("simulated prefetch failure"))
        streamer._pending[0] = bad_future

        with pytest.raises(RuntimeError, match="simulated prefetch failure"):
            strategy.deactivate()

        # Even though we raised, cleanup completed.
        assert not streamer._hooks
        assert streamer._executor is None

        strategy.deactivate()


# ---------------------------------------------------------------------------
# Constructor leaves no GPU residency (cache_bytes is final, no pool yet)
# ---------------------------------------------------------------------------


class TestConstructedStateIsInactive:
    """Verifies the 'constructed but not active' state has no GPU
    footprint — the payoff of the SlotOwnership-based composition.
    Without it, non-block siblings (embed, head, norms) would sit on
    target_device permanently, defeating ModelCache eviction."""

    @CUDA
    def test_constructed_has_no_params_on_target_device(self) -> None:
        m = _make_block_model(num_blocks=4, width=8)
        target = torch.device("cuda")
        strategy = ModelOffloader(
            m, target, layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            for p in m.parameters():
                assert p.device != target, (
                    f"constructed state leaked GPU residency: {p.shape}@{p.device}"
                )
        finally:
            strategy.deactivate()

    def test_block_only_model_has_no_non_block_pinned(self) -> None:
        # Edge case: model whose only top-level child IS the block list.
        # No non-block PinnedWeights in components; cache_bytes from blocks only.
        class BlockOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False) for _ in range(4)]
                )

        m = BlockOnly()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader(
            m, torch.device("cpu"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            # No PinnedWeights component, just TrainableWeights + StreamedWeights.
            non_block_components = [
                c for c in strategy._components if isinstance(c, PinnedWeights)
            ]
            assert non_block_components == []
            assert strategy.cache_bytes > 0  # block bytes only
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Buffer-only non-block module (e.g., a RoPE table)
# ---------------------------------------------------------------------------


class TestBufferOnlyNonBlock:
    @CUDA
    def test_buffer_only_non_block_module(self) -> None:
        class RopeTable(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.randn(8, 4))

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.rope = RopeTable()
                self.transformer_blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False) for _ in range(4)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        target = torch.device("cuda")
        strategy = ModelOffloader(
            m, target, layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            assert m.rope.table.is_pinned()
            strategy.activate()
            assert m.rope.table.is_cuda
            strategy.deactivate()
            assert m.rope.table.is_pinned()
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Cross-region tied-weight detection
# ---------------------------------------------------------------------------


class TestCrossRegionTiedDetection:
    def test_cross_block_tied_raises(self) -> None:
        shared = torch.randn(8, 8)
        block_0 = nn.Linear(8, 8, bias=False)
        block_1 = nn.Linear(8, 8, bias=False)
        block_0.weight = nn.Parameter(shared, requires_grad=False)
        block_1.weight = nn.Parameter(shared, requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([block_0, block_1])

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="streamed regions|tied"):
            ModelOffloader(
                m, torch.device("cpu"),
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )

    def test_block_to_non_block_tied_raises(self) -> None:
        shared = torch.randn(4, 4)
        block_0 = nn.Linear(4, 4, bias=False)
        block_0.weight = nn.Parameter(shared, requires_grad=False)
        head = nn.Linear(4, 4, bias=False)
        head.weight = nn.Parameter(shared, requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [block_0, nn.Linear(4, 4, bias=False)]
                )
                self.head = head

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="streamed regions|tied"):
            ModelOffloader(
                m, torch.device("cpu"),
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )

    def test_mixed_trainable_frozen_cross_region_tied_raises(self) -> None:
        shared = torch.randn(4, 4)
        block_0 = nn.Linear(4, 4, bias=False)
        block_0.weight = nn.Parameter(shared, requires_grad=True)
        head = nn.Linear(4, 4, bias=False)
        head.weight = nn.Parameter(shared, requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [block_0, nn.Linear(4, 4, bias=False)]
                )
                self.head = head

        m = M()
        for p in m.transformer_blocks[1].parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="streamed regions|tied"):
            ModelOffloader(
                m, torch.device("cpu"),
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )

    def test_intra_block_tied_params_raises(self) -> None:
        shared = torch.randn(8, 8)

        class TiedBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn_q = nn.Linear(8, 8, bias=False)
                self.attn_k = nn.Linear(8, 8, bias=False)
                self.attn_q.weight = nn.Parameter(shared, requires_grad=False)
                self.attn_k.weight = nn.Parameter(shared, requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [TiedBlock(), nn.Linear(8, 8, bias=False)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="intra-block tied"):
            ModelOffloader(
                m, torch.device("cpu"),
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )

    def test_cross_region_tied_buffers_raises(self) -> None:
        shared = torch.randn(8)
        view_block = shared.view(8)
        view_aux = shared.view(8)
        assert id(view_block) != id(view_aux)
        assert view_block.data_ptr() == view_aux.data_ptr()

        class BlockWithTiedBuf(nn.Module):
            def __init__(self, buf):
                super().__init__()
                self.register_buffer("table", buf)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [BlockWithTiedBuf(view_block), nn.Linear(4, 4, bias=False)]
                )
                self.aux = nn.Module()
                self.aux.register_buffer("alias", view_aux)

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="tied buffers across"):
            ModelOffloader(
                m, torch.device("cpu"),
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )

    def test_same_buffer_object_cross_region_raises(self) -> None:
        # Bug 3 regression: the SAME Python buffer object registered
        # at both a block path AND a non-block path. Previously this
        # was missed (id-based classification put both in block
        # region). Slot-ownership classification now catches it.
        shared_buf = torch.randn(8)

        class BlockWithBuf(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", shared_buf)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [BlockWithBuf(), nn.Linear(4, 4, bias=False)]
                )
                # Same Python buffer object also registered at non-block path.
                self.register_buffer("aux", shared_buf)

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="tied buffers across"):
            ModelOffloader(
                m, torch.device("cpu"),
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )

    def test_intra_block_tied_buffers_raises(self) -> None:
        shared = torch.randn(8)
        view_a = shared.view(8)
        view_b = shared.view(8)

        class TiedBufBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf_a", view_a)
                self.register_buffer("buf_b", view_b)
                self.weight = nn.Parameter(torch.randn(2), requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [TiedBufBlock(), nn.Linear(4, 4, bias=False)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="intra-block tied buffers"):
            ModelOffloader(
                m, torch.device("cpu"),
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )

    def test_non_block_internal_tied_works(self) -> None:
        # Tied embed↔head WITHIN non-block region: PinnedWeights handles
        # this via storage-key dedup. Should not raise.
        embed = nn.Embedding(16, 8)
        head = nn.Linear(8, 16, bias=False)
        head.weight = embed.weight  # standard tie_weights() pattern

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = embed
                self.transformer_blocks = nn.ModuleList(
                    [nn.Linear(8, 8, bias=False) for _ in range(4)]
                )
                self.head = head

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader(
            m, torch.device("cpu"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            assert m.embed.weight is m.head.weight
            non_block = next(
                c for c in strategy._components if isinstance(c, PinnedWeights)
            )
            assert len(non_block.slots) == 1  # tie deduped
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Direct-parent state handling — via SlotOwnership skip filter
# ---------------------------------------------------------------------------


class TestDirectParentStateHandled:
    def test_direct_frozen_param_on_root_is_pinned(self) -> None:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale_shift = nn.Parameter(torch.randn(4), requires_grad=False)
                self.transformer_blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False) for _ in range(4)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader(
            m, torch.device("cpu"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            assert m.scale_shift.is_pinned()
        finally:
            strategy.deactivate()

    def test_nested_layers_attr_with_direct_root_param(self) -> None:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.root_param = nn.Parameter(torch.randn(4), requires_grad=False)
                self.encoder = nn.Module()
                self.encoder.blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False) for _ in range(4)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader(
            m, torch.device("cpu"),
            layers_attr="encoder.blocks", blocks_to_swap=2,
        )
        try:
            assert m.root_param.is_pinned()
        finally:
            strategy.deactivate()

    def test_direct_buffer_on_root_is_pinned(self) -> None:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.randn(8))
                self.transformer_blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False) for _ in range(4)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader(
            m, torch.device("cpu"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            assert m.table.is_pinned()
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Block-internal buffers
# ---------------------------------------------------------------------------


class TestBlockBuffersPinned:
    def test_block_buffer_clone_is_pinned(self) -> None:
        class BlockWithBuffer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)
                self.register_buffer("table", torch.randn(8))

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [BlockWithBuffer() for _ in range(4)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = ModelOffloader(
            m, torch.device("cpu"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            for block in m.transformer_blocks:
                assert block.table.is_pinned(), (
                    "block buffer must be pinned for honest cache_bytes "
                    "and to avoid silently-synchronous H2D copies"
                )
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# _BlockPinnedStore activate_pool idempotency
# ---------------------------------------------------------------------------


class TestActivatePoolIdempotency:
    @CUDA
    def test_same_config_idempotent(self) -> None:
        m = _make_block_model()
        store = _BlockPinnedStore(list(m.transformer_blocks))
        store.activate_pool(2, torch.device("cuda"))
        pool_first = store._pool
        store.activate_pool(2, torch.device("cuda"))
        assert store._pool is pool_first

    @CUDA
    def test_mismatched_config_raises(self) -> None:
        m = _make_block_model()
        store = _BlockPinnedStore(list(m.transformer_blocks))
        store.activate_pool(2, torch.device("cuda"))
        with pytest.raises(ValueError, match="already activated"):
            store.activate_pool(3, torch.device("cuda"))
        with pytest.raises(ValueError, match="already activated"):
            store.activate_pool(2, torch.device("cpu"))


# ---------------------------------------------------------------------------
# Layout-signature check rejects heterogeneous block layouts
# ---------------------------------------------------------------------------


class TestBlockLayoutSignature:
    """Block 0 is the pool template; later blocks copy raw bytes into
    its slot. ``Tensor.copy_`` silently casts dtype and silently
    broadcasts compatible shapes, so mismatches that don't trip the
    copy_ shape check would silently corrupt forward. The constructor's
    layout-signature check rejects them up front."""

    def test_shape_mismatch_raises(self) -> None:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False), nn.Linear(8, 8, bias=False)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="layout differs"):
            StreamedWeights(
                blocks=list(m.blocks),
                target_device=torch.device("cpu"),
                blocks_to_swap=1,
            )

    def test_dtype_mismatch_raises(self) -> None:
        # Same shape, different dtype: Tensor.copy_ would silently
        # cast — the silent corruption surface this check exists for.
        b0 = nn.Linear(4, 4, bias=False).to(torch.float16)
        b1 = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
        for b in (b0, b1):
            for p in b.parameters():
                p.requires_grad = False
        with pytest.raises(ValueError, match="layout differs"):
            StreamedWeights(
                blocks=[b0, b1],
                target_device=torch.device("cpu"),
                blocks_to_swap=1,
            )

    def test_param_name_mismatch_raises(self) -> None:
        # Different param names per block — would be a KeyError later;
        # the check catches it cleanly at construction.
        class A(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = nn.Parameter(torch.randn(4, 4), requires_grad=False)

        class B(nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = nn.Parameter(torch.randn(4, 4), requires_grad=False)

        with pytest.raises(ValueError, match="layout differs"):
            StreamedWeights(
                blocks=[A(), B()],
                target_device=torch.device("cpu"),
                blocks_to_swap=1,
            )

    def test_failure_leaves_model_unpinned_and_unmutated(self) -> None:
        # Strong-exception-safety: the validator runs in pass 1
        # (collect specs) before pass 2 (pin) and pass 3 (apply slot
        # mutations). On a layout mismatch, the user's Parameter
        # objects must be the same identities and not pinned —
        # neither pin_memory() nor _parameters[leaf] = ... fires.
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False), nn.Linear(8, 8, bias=False)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False

        original_params = [b.weight for b in m.blocks]
        original_pinned = [b.weight.is_pinned() for b in m.blocks]

        with pytest.raises(ValueError, match="layout differs"):
            StreamedWeights(
                blocks=list(m.blocks),
                target_device=torch.device("cpu"),
                blocks_to_swap=1,
            )

        for block, orig_p, orig_pin in zip(
            m.blocks, original_params, original_pinned, strict=True,
        ):
            assert block.weight is orig_p, (
                "slot identity mutated despite pre-pin validation failure"
            )
            assert block.weight.is_pinned() == orig_pin, (
                "param was pinned despite pre-pin validation failure"
            )



# ---------------------------------------------------------------------------
# Multi-component cleanup ordering (ExitStack semantics)
# ---------------------------------------------------------------------------


class TestMultiComponentCleanup:
    @CUDA
    def test_trainable_move_failure_still_runs_other_deactivates(self) -> None:
        # ExitStack continues unwinding callbacks even when one raises.
        # If TrainableWeights's deactivate raises, StreamedWeights (earlier
        # in unwind order) and non_block PinnedWeights (later in unwind)
        # still get their deactivate called.
        from unittest.mock import patch

        m = _make_block_model()
        strategy = ModelOffloader(
            m, torch.device("cuda"),
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.activate()
            assert m.embed.weight.is_cuda  # type: ignore[union-attr]

            with patch.object(
                TrainableWeights, "_move",
                side_effect=RuntimeError("simulated trainable move failure"),
            ), pytest.raises(RuntimeError, match="simulated trainable move failure"):
                strategy.deactivate()

            # Despite the trainable-move failure, non_block was
            # deactivated (slots back to pinned CPU) — proves
            # ExitStack continued past the raising callback.
            assert m.embed.weight.is_pinned()  # type: ignore[union-attr]
            assert strategy._teardown_stack is None
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# TrainableWeights (component-level tests)
# ---------------------------------------------------------------------------


class TestTrainableWeights:
    def test_cache_bytes_is_zero(self) -> None:
        m = _make_block_model()
        mover = TrainableWeights(m, torch.device("cpu"))
        assert mover.cache_bytes == 0
        mover.deactivate()

    def test_activate_and_deactivate_noop_when_no_trainable(self) -> None:
        m = _make_block_model()  # all frozen
        mover = TrainableWeights(m, torch.device("cpu"))
        try:
            mover.activate()
            mover.deactivate()
        finally:
            mover.deactivate()

    @CUDA
    def test_moves_trainable_param_to_target_device_on_activate(self) -> None:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora = nn.Parameter(torch.randn(4))  # trainable

        m = M()
        target = torch.device("cuda")
        mover = TrainableWeights(m, target)
        try:
            mover.activate()
            assert m.lora.is_cuda
            mover.deactivate()
            assert not m.lora.is_cuda  # back to CPU
        finally:
            mover.deactivate()

    def test_deactivate_idempotent(self) -> None:
        m = _make_block_model()
        mover = TrainableWeights(m, torch.device("cpu"))
        mover.deactivate()
        mover.deactivate()


# ---------------------------------------------------------------------------
# SlotOwnership-based filter survives slot mutation
# ---------------------------------------------------------------------------


class TestSlotOwnershipFilter:
    """The SlotOwnership skip filter is the design fix that decouples
    construction order. It identifies slots by (id(parent), leaf, kind)
    so PinnedWeights's skip check still matches even after a streamer
    has swapped the Parameter object at that slot."""

    def test_streamed_weights_slot_filter_is_slot_ownership_set(self) -> None:
        m = _make_block_model()
        streamer = StreamedWeights(
            blocks=list(m.transformer_blocks),
            target_device=torch.device("cpu"),
            blocks_to_swap=2,
        )
        try:
            sf = streamer.slot_filter
            assert isinstance(sf, frozenset)
            for s in sf:
                assert isinstance(s, SlotOwnership)
                assert s.kind in ("param", "buffer")
        finally:
            streamer.deactivate()

    def test_pinned_weights_skips_slots_after_streamer_swapped_them(self) -> None:
        # Constructor order independence: build PinnedWeights AFTER
        # the streamer has already mutated slots. With id()-based
        # filter, this would fail (the original Parameter ids no
        # longer match what's in the slots). With SlotOwnership it
        # works because (parent, leaf) is stable.
        m = _make_block_model()
        # Build the streamer first — this swaps block slots to pinned
        # cpu_params (different Python objects than the originals).
        streamer = StreamedWeights(
            blocks=list(m.transformer_blocks),
            target_device=torch.device("cpu"),
            blocks_to_swap=2,
        )
        try:
            skip_slots = set(streamer.slot_filter)

            # Build PinnedWeights AFTER the streamer mutated slots.
            # The skip filter must still correctly exclude block-owned
            # slots, even though the Parameter objects changed.
            non_block = PinnedWeights(
                m, torch.device("cpu"), skip_slots=skip_slots,
            )
            try:
                # Non-block slots (embed, head) are pinned by
                # PinnedWeights; block slots are skipped (already
                # owned by streamer).
                slots_managed_by_pinned = {
                    SlotOwnership(id(parent), leaf, "param")
                    for _, locs in non_block.slots
                    for parent, leaf in locs
                }
                # Block-owned slots NOT in the PinnedWeights set.
                for s in skip_slots:
                    assert s not in slots_managed_by_pinned, (
                        f"PinnedWeights tried to manage block-owned slot {s}"
                    )
                # Non-block slots present.
                non_block_slot = SlotOwnership(id(m.embed), "weight", "param")
                assert non_block_slot in slots_managed_by_pinned
            finally:
                non_block.deactivate()
        finally:
            streamer.deactivate()


# ---------------------------------------------------------------------------
# detect_streaming_region_ties (free-function, multi-group)
# ---------------------------------------------------------------------------


class TestDetectStreamingRegionTies:
    def test_passes_for_clean_model(self) -> None:
        m = _make_block_model()
        # Should not raise.
        detect_streaming_region_ties(m, [list(m.transformer_blocks)])

    def test_multi_group_cross_group_tied_raises(self) -> None:
        # Same Parameter object appearing in two different groups —
        # this must be rejected (slot-local streaming can't preserve
        # cross-group sharing).
        shared = torch.randn(4, 4)
        block_a = nn.Linear(4, 4, bias=False)
        block_b = nn.Linear(4, 4, bias=False)
        block_a.weight = nn.Parameter(shared, requires_grad=False)
        block_b.weight = nn.Parameter(shared, requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.group_a = nn.ModuleList([block_a, nn.Linear(4, 4, bias=False)])
                self.group_b = nn.ModuleList([block_b, nn.Linear(4, 4, bias=False)])

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="tied|streamed regions"):
            detect_streaming_region_ties(
                m, [list(m.group_a), list(m.group_b)],
            )


# ---------------------------------------------------------------------------
# Composer-driven trainable partitioning (contract guards + tie detection)
# ---------------------------------------------------------------------------


class TestStreamedWeightsContractGuard:
    """StreamedWeights is frozen-only by mechanism (slot replacement
    breaks Parameter identity). Direct callers must partition trainables
    via skip_slots; the composer does this automatically."""

    def test_direct_unskipped_trainable_raises(self) -> None:
        # Build a trainable block. No skip_slots → contract guard fires
        # at construction. Fail loudly rather than silently freeze the
        # param.
        block = nn.Linear(4, 4, bias=False)  # default requires_grad=True
        with pytest.raises(ValueError, match="cannot manage trainable slot"):
            StreamedWeights(
                blocks=[block, nn.Linear(4, 4, bias=False)],
                target_device=torch.device("cpu"),
                blocks_to_swap=1,
            )

    def test_direct_skipped_trainable_constructs(self) -> None:
        # With the trainable slot in skip_slots, construction succeeds
        # and the slot is excluded from slot_filter.
        from torch_offload.slots import iter_param_slots

        block_0 = nn.Linear(4, 4, bias=False)  # trainable
        block_1 = nn.Linear(4, 4, bias=False)  # trainable
        # Snapshot trainable slots before StreamedWeights construction.
        trainable_slots = {
            s.slot for s in iter_param_slots(block_0) if s.param.requires_grad
        } | {
            s.slot for s in iter_param_slots(block_1) if s.param.requires_grad
        }
        # No frozen content remains, so the streamer's pinning walk
        # produces empty buffers but no contract violation.
        streamer = StreamedWeights(
            blocks=[block_0, block_1],
            target_device=torch.device("cpu"),
            blocks_to_swap=1,
            skip_slots=trainable_slots,
        )
        assert streamer.slot_filter.isdisjoint(trainable_slots)

    def test_direct_skipped_buffers_are_not_pinned_or_owned(self) -> None:
        class BufferBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4), requires_grad=False)
                self.register_buffer("table", torch.randn(8))

        blocks = [BufferBlock(), BufferBlock()]
        buffer_slots = {
            s.slot for block in blocks for s in iter_buffer_slots(block)
        }
        buffer_ptrs = [block.table.data_ptr() for block in blocks]

        streamer = StreamedWeights(
            blocks=blocks,
            target_device=torch.device("cpu"),
            blocks_to_swap=1,
            skip_slots=buffer_slots,
        )

        assert streamer.slot_filter.isdisjoint(buffer_slots)
        assert [block.table.data_ptr() for block in blocks] == buffer_ptrs
        assert all(not block.table.is_pinned() for block in blocks)


class TestMixedGradTieDetection:
    """detect_streaming_region_ties must catch mixed-grad ties anywhere
    — across regions, intra-block, or intra-non-block. Slot replacement
    (frozen) and storage swap (trainable) cannot cohabit a tied
    storage."""

    def test_intra_non_block_mixed_grad_tie_raises(self) -> None:
        # Two distinct Parameter objects sharing storage, both in the
        # non-block region, with mixed grad. Slot replacement on the
        # frozen alias would diverge from the storage-swap-managed
        # trainable alias on activate.
        shared = torch.randn(4, 4)
        a = nn.Parameter(shared, requires_grad=True)
        b = nn.Parameter(shared, requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)]
                )
                self.alias_a = nn.Module()
                self.alias_b = nn.Module()
                self.alias_a.weight = a
                self.alias_b.weight = b

        m = M()
        for p in m.transformer_blocks.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="trainable and frozen"):
            ModelOffloader(
                m, torch.device("cpu"),
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )

    def test_all_trainable_distinct_parameter_tie_raises(self) -> None:
        # Two distinct Parameter objects sharing storage, both trainable.
        # TrainableWeights walks model.parameters() (deduped by id(p)) and
        # would move each Parameter independently, breaking the storage
        # alias on GPU. Reject upfront.
        shared = torch.randn(4, 4)
        a = nn.Parameter(shared, requires_grad=True)
        b = nn.Parameter(shared, requires_grad=True)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)]
                )
                self.alias_a = nn.Module()
                self.alias_b = nn.Module()
                self.alias_a.weight = a
                self.alias_b.weight = b

        m = M()
        for p in m.transformer_blocks.parameters():
            p.requires_grad = False
        with pytest.raises(
            ValueError, match="distinct Parameter objects|tie_weights",
        ):
            ModelOffloader(
                m, torch.device("cpu"),
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )

    def test_all_trainable_same_parameter_tie_constructs(self) -> None:
        shared = nn.Parameter(torch.randn(4, 4), requires_grad=True)

        class TiedTrainableBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(4, 4, bias=False)
                self.base.weight.requires_grad = False
                self.a = shared
                self.b = shared

        class FrozenBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(4, 4, bias=False)
                self.base.weight.requires_grad = False

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [TiedTrainableBlock(), FrozenBlock()]
                )

        m = M()

        strategy = ModelOffloader(
            m, torch.device("cpu"),
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )

        assert m.transformer_blocks[0]._parameters["a"] is shared
        assert m.transformer_blocks[0]._parameters["b"] is shared
        strategy.deactivate()

    def test_mixed_grad_cross_region_reports_grad_cause(self) -> None:
        # When a tie is BOTH cross-region AND mixed-grad, the user
        # should see the mixed-grad cause (the more specific one) — the
        # cross-region recovery advice (whole-model PinnedWeights)
        # wouldn't fix mixed-grad anyway.
        shared = torch.randn(4, 4)
        block_0 = nn.Linear(4, 4, bias=False)
        block_0.weight = nn.Parameter(shared, requires_grad=True)
        head = nn.Linear(4, 4, bias=False)
        head.weight = nn.Parameter(shared, requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [block_0, nn.Linear(4, 4, bias=False)]
                )
                self.head = head

        m = M()
        for p in m.transformer_blocks[1].parameters():
            p.requires_grad = False
        with pytest.raises(
            ValueError, match="trainable and frozen",
        ):
            ModelOffloader(
                m, torch.device("cpu"),
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )

    def test_intra_block_mixed_grad_tie_raises(self) -> None:
        # Two distinct Parameter objects sharing storage, both inside
        # the same block, with mixed grad. The intra-block tie check
        # already fires for distinct slots; this test ensures the
        # mixed-grad message takes precedence (or at minimum, that the
        # composer rejects).
        shared = torch.randn(4, 4)

        class TiedBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = nn.Linear(4, 4, bias=False)
                self.k = nn.Linear(4, 4, bias=False)
                self.q.weight = nn.Parameter(shared, requires_grad=True)
                self.k.weight = nn.Parameter(shared, requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [TiedBlock(), nn.Linear(4, 4, bias=False)]
                )

        m = M()
        for p in m.transformer_blocks[1].parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="trainable and frozen|intra-block tied"):
            ModelOffloader(
                m, torch.device("cpu"),
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )


class TestLoRAInBlockRouting:
    """LoRA-shaped models: blocks contain frozen base layers plus
    trainable adapter layers. The composer must route the base to
    StreamedWeights and the adapters to TrainableWeights; neither
    strategy's contract guard should fire on a well-formed LoRA model.
    """

    def _make_lora_block(self) -> nn.Module:
        """One LoRA-wrapped layer: frozen base.weight + trainable
        lora_a/lora_b weights."""

        class LoraBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(4, 4, bias=False)
                self.lora_a = nn.Linear(4, 2, bias=False)
                self.lora_b = nn.Linear(2, 4, bias=False)
                self.base.weight.requires_grad = False
                # lora_a/lora_b stay trainable (default)

        return LoraBlock()

    def test_composer_routes_lora_to_trainable_mover(self) -> None:
        class M(nn.Module):
            def __init__(self, blocks):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(blocks)

        m = M([self._make_lora_block() for _ in range(2)])
        strat = ModelOffloader(
            m, torch.device("cpu"),
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        try:
            # Strategy composes a TrainableWeights for the LoRA params.
            assert any(
                isinstance(c, TrainableWeights) for c in strat._components
            )
            # Each StreamedWeights's slot_filter only contains frozen
            # base.weight slots; lora_a/lora_b are skipped.
            streamers = [
                c for c in strat._components if isinstance(c, StreamedWeights)
            ]
            assert len(streamers) == 1
            streamer = streamers[0]
            # Walk the model and confirm: lora slot ownerships are NOT in
            # streamer's slot_filter; base.weight slot ownerships ARE.
            from torch_offload.slots import iter_param_slots
            for s in iter_param_slots(m):
                if s.param.requires_grad:
                    assert s.slot not in streamer.slot_filter, (
                        f"trainable slot {s.name} leaked into streamer's "
                        f"slot_filter"
                    )
                elif "transformer_blocks" in s.name and "base.weight" in s.name:
                    assert s.slot in streamer.slot_filter, (
                        f"frozen base slot {s.name} missing from "
                        f"streamer's slot_filter"
                    )
        finally:
            strat.deactivate()

    def test_composer_partitions_skip_slots_correctly(self) -> None:
        # Through-test: the composer's PinnedWeights receives skip_slots
        # = block_slots ∪ trainable_slots. Verify by checking that no
        # trainable param slot appears in PinnedWeights' managed slots
        # and no block-internal slot does either.
        from torch_offload.slots import iter_param_slots

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)]
                )
                # Non-block content: one frozen + one trainable.
                self.frozen_head = nn.Linear(4, 4, bias=False)
                self.trainable_bias = nn.Parameter(torch.zeros(4))

        m = M()
        for p in m.transformer_blocks.parameters():
            p.requires_grad = False
        m.frozen_head.weight.requires_grad = False
        # m.trainable_bias stays trainable

        strat = ModelOffloader(
            m, torch.device("cpu"),
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        try:
            pinned = next(
                c for c in strat._components if isinstance(c, PinnedWeights)
            )
            # PinnedWeights manages frozen_head.weight only — block content
            # routed to StreamedWeights, trainable_bias to TrainableWeights.
            managed_slot_ids = {
                (id(parent), leaf)
                for _buf, locs in pinned.slots
                for parent, leaf in locs
            }
            for s in iter_param_slots(m):
                key = (id(s.parent), s.leaf)
                if s.name == "frozen_head.weight":
                    assert key in managed_slot_ids
                else:
                    assert key not in managed_slot_ids, (
                        f"slot {s.name} (requires_grad={s.param.requires_grad}) "
                        f"leaked into PinnedWeights"
                    )
        finally:
            strat.deactivate()
