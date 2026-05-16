"""Tests for the block-streaming machinery in ``torch_offload``.

Covers ``ModelOffloader`` (the public composite),
``StreamedWeights`` (the per-block-list primitive),
``TrainableWeights`` (the trainable-param component),
and the cross-region tied-weight detector.

CUDA-only tests gate on availability. CPU activation is pass-through
over the host-backed pinned state.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future

import pytest
import torch
import torch.utils.checkpoint
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


def _make_trainable_block_model(num_blocks: int = 4, width: int = 8) -> nn.Module:
    """LoRA-shaped training model: trainable embed/head wrapping
    a frozen streamed block list.

    Forward accepts ``use_checkpoint=`` so a single model can be
    driven both ways under the training tests — with checkpointing
    (correct under streaming) and without (expected to raise on
    backward via autograd's saved-tensor version check).
    """

    class TrainableBlockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(width, width, bias=False)
            self.transformer_blocks = nn.ModuleList(
                nn.Linear(width, width, bias=False) for _ in range(num_blocks)
            )
            self.head = nn.Linear(width, width, bias=False)

        def forward(
            self, x: torch.Tensor, *, use_checkpoint: bool = False
        ) -> torch.Tensor:
            x = self.embed(x)
            for block in self.transformer_blocks:
                if use_checkpoint:
                    x = torch.utils.checkpoint.checkpoint(
                        block, x, use_reentrant=False,
                    )
                else:
                    x = block(x)
            return self.head(x)

    m = TrainableBlockModel()
    for p in m.transformer_blocks.parameters():
        p.requires_grad = False
    return m


# ---------------------------------------------------------------------------
# ModelStrategy conformance
# ---------------------------------------------------------------------------


class TestModelStrategyConformance:
    def test_isinstance_runtime_check(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            assert isinstance(strategy, ModelStrategy)
        finally:
            strategy.deactivate()

    def test_has_lifecycle_methods(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m,
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
            m,
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
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.activate("cuda")
            assert strategy.model is m
        finally:
            strategy.deactivate()

    @CUDA
    def test_activate_canonicalizes_bare_cuda_device(self) -> None:
        m = _make_block_model()
        expected = torch.device("cuda", torch.cuda.current_device())
        strategy = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.activate("cuda")
            assert strategy._active_device == expected
            assert strategy._streamers[0]._active_device == expected
        finally:
            strategy.deactivate()

    def test_activate_cpu_uses_host_backed_weights_without_streaming(self) -> None:
        torch.manual_seed(42)
        m_eager = _make_block_model(num_blocks=4, width=8)
        m_off = _make_block_model(num_blocks=4, width=8)
        m_off.load_state_dict(m_eager.state_dict())

        x = torch.randn(2, 8)
        with torch.no_grad():
            expected = m_eager(x)

        strategy = ModelOffloader(
            m_off,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            pinned_block_params = [
                block.weight for block in m_off.transformer_blocks
            ]
            with strategy.use("cpu") as cpu_model:
                assert strategy._active_device == torch.device("cpu")
                assert all(
                    s._active_device == torch.device("cpu")
                    for s in strategy._streamers
                )
                assert all(s._executor is None for s in strategy._streamers)
                assert all(
                    block.weight is pinned
                    for block, pinned in zip(
                        m_off.transformer_blocks,
                        pinned_block_params,
                        strict=True,
                    )
                )
                with torch.no_grad():
                    got = cpu_model(x)

            torch.testing.assert_close(got, expected)
            for p in m_off.parameters():
                assert p.device == torch.device("cpu")
                assert p.is_pinned()
        finally:
            strategy.deactivate()

    @CUDA
    def test_activate_brings_non_block_to_gpu(self) -> None:
        m = _make_block_model()
        target = torch.device("cuda")
        strategy = ModelOffloader(
            m, layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.activate("cuda")
            assert m.embed.weight.is_cuda
            assert m.head.weight.is_cuda
        finally:
            strategy.deactivate()

    @CUDA
    def test_deactivate_returns_non_block_to_pinned(self) -> None:
        m = _make_block_model()
        target = torch.device("cuda")
        strategy = ModelOffloader(
            m, layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.activate("cuda")
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
            m, layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            cache_bytes = strategy.cache_bytes
            strategy.activate("cuda")
            strategy.deactivate()
            assert strategy.cache_bytes == cache_bytes
            strategy.activate("cuda")
            strategy.deactivate()
        finally:
            strategy.deactivate()

    def test_deactivate_when_not_active_is_noop(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.deactivate()  # no error, never activated
            strategy.deactivate()  # still no error
        finally:
            strategy.deactivate()

    def test_double_activate_raises_before_component_activation(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.activate("cpu")

            def fail_activate(device: torch.device | str | None = None) -> None:
                del device
                raise AssertionError("component activation should not run")

            monkeypatch.setattr(strategy._components[0], "activate", fail_activate)
            with pytest.raises(RuntimeError, match=r"already.*active"):
                strategy.activate("cpu")
        finally:
            strategy.deactivate()


class TestStreamedWeightsBackendActivation:
    def test_direct_cpu_activation_uses_host_backed_weights(self) -> None:
        torch.manual_seed(42)
        m_eager = _make_block_model(num_blocks=4, width=8)
        m = _make_block_model(num_blocks=4, width=8)
        m.load_state_dict(m_eager.state_dict())

        x = torch.randn(2, 8)
        with torch.no_grad():
            expected = m_eager(x)

        streamer = StreamedWeights(
            blocks=list(m.transformer_blocks),
            blocks_to_swap=2,
        )
        try:
            pinned_params = [block.weight for block in m.transformer_blocks]
            with streamer.use("cpu"):
                assert streamer._active_device == torch.device("cpu")
                assert streamer._executor is None
                assert all(
                    block.weight is pinned
                    for block, pinned in zip(
                        m.transformer_blocks, pinned_params, strict=True,
                    )
                )
                assert all(
                    len(block._forward_pre_hooks) == 0
                    for block in m.transformer_blocks
                )
                with torch.no_grad():
                    got = m(x)

            torch.testing.assert_close(got, expected)
            for block in m.transformer_blocks:
                assert block.weight.device == torch.device("cpu")
                assert block.weight.is_pinned()
        finally:
            streamer.deactivate()

    def test_direct_cpu_trainable_step_preserves_updates(self) -> None:
        torch.manual_seed(0)
        blocks = nn.ModuleList(
            [nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)]
        )
        param_ids = {i: id(block.weight) for i, block in enumerate(blocks)}
        before = {i: block.weight.detach().clone() for i, block in enumerate(blocks)}
        optimizer = torch.optim.SGD(blocks.parameters(), lr=0.1)

        streamer = StreamedWeights(
            blocks=list(blocks),
            blocks_to_swap=1,
        )
        try:
            pinned_data_ptrs = {
                i: block.weight.data_ptr() for i, block in enumerate(blocks)
            }
            with streamer.use("cpu"):
                assert pinned_data_ptrs == {
                    i: block.weight.data_ptr()
                    for i, block in enumerate(blocks)
                }
                x = torch.randn(2, 4)
                target = torch.randn(2, 4)
                out = x
                for block in blocks:
                    out = block(out)
                ((out - target) ** 2).mean().backward()
                with streamer.optimizer_step():
                    optimizer.step()
                active_after = {
                    i: block.weight.detach().clone()
                    for i, block in enumerate(blocks)
                }

            assert param_ids == {i: id(block.weight) for i, block in enumerate(blocks)}
            assert any(
                not torch.allclose(active_after[i], before[i])
                for i in active_after
            )
            for i, block in enumerate(blocks):
                torch.testing.assert_close(block.weight, active_after[i])
                assert block.weight.device == torch.device("cpu")
                assert block.weight.is_pinned()
        finally:
            streamer.deactivate()

    def test_direct_cpu_double_activate_raises(self) -> None:
        m = _make_block_model(num_blocks=2)
        streamer = StreamedWeights(
            blocks=list(m.transformer_blocks),
            blocks_to_swap=1,
        )
        try:
            streamer.activate("cpu")
            with pytest.raises(RuntimeError, match="already.*active"):
                streamer.activate("cpu")
        finally:
            streamer.deactivate()


# ---------------------------------------------------------------------------
# Cleanup contract: deactivate restores CPU state, drop refs to free
# ---------------------------------------------------------------------------


class TestCleanup:
    @CUDA
    def test_deactivate_restores_cpu_state(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strategy.activate("cuda")
        strategy.deactivate()
        # Model is back in CPU/pinned state — usable, just without the
        # strategy's GPU streaming.
        for p in m.parameters():
            assert not p.is_cuda

    def test_deactivate_is_idempotent(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strategy.deactivate()
        strategy.deactivate()  # no error

    @CUDA
    def test_deactivate_consumes_teardown_stack(self) -> None:
        m = _make_block_model()
        strategy = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strategy.activate("cuda")
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
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strategy.activate("cuda")  # installs hooks; no deactivate
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
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strategy.activate("cuda")
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
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.activate("cuda")
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
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strategy.activate("cuda")
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
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
            prefetch_count=2, cyclic=True,
        )
        streamer: StreamedWeights = strategy._streamers[0]

        with strategy.use("cuda"):
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
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
            prefetch_count=2, cyclic=False,
        )
        streamer: StreamedWeights = strategy._streamers[0]

        with strategy.use("cuda"):
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
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
            prefetch_count=1, cyclic=True,
        )
        streamer: StreamedWeights = strategy._streamers[0]

        with strategy.use("cuda"):
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
            m_off,
            layers_attr="transformer_blocks", blocks_to_swap=1,
            prefetch_count=2, cyclic=True,
        )
        try:
            with strategy.use("cuda"):
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
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
            prefetch_count=1, cyclic=True,
        )
        streamer: StreamedWeights = strategy._streamers[0]

        with strategy.use("cuda"):
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
            m_off,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.activate("cuda")
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
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            x = torch.randn(2, 8, device="cuda")
            strategy.activate("cuda")
            with torch.no_grad():
                first = m(x)
            torch.cuda.synchronize()
            strategy.deactivate()

            strategy.activate("cuda")
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
                m,
                layers_attr="transformer_blocks", blocks_to_swap=4,  # equal to num_blocks
            )

    def test_blocks_to_swap_must_be_non_negative(self) -> None:
        m = _make_block_model(num_blocks=4)
        with pytest.raises(ValueError, match="blocks_to_swap"):
            ModelOffloader(
                m,
                layers_attr="transformer_blocks", blocks_to_swap=-1,
            )

    def test_prefetch_count_must_be_non_negative(self) -> None:
        m = _make_block_model(num_blocks=4)
        with pytest.raises(ValueError, match="prefetch_count"):
            ModelOffloader(
                m,
                layers_attr="transformer_blocks", blocks_to_swap=2,
                prefetch_count=-1,
            )

    def test_empty_layers_attr_raises(self) -> None:
        m = _make_block_model(num_blocks=4)
        with pytest.raises(ValueError, match="at least one path"):
            ModelOffloader(
                m,
                layers_attr=[], blocks_to_swap=2,
            )

    def test_layers_attr_resolving_to_non_modulelist_raises(self) -> None:
        m = _make_block_model(num_blocks=4)
        with pytest.raises(TypeError, match="nn.ModuleList"):
            ModelOffloader(
                m,
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
                m, layers_attr="transformer_blocks", blocks_to_swap=2,
            )

        cache = ModelCache(max_cache_bytes=10_000_000)
        spec = ModelSpec(key="xformer", estimated_cache_bytes=1024, factory=factory)

        with cache.use(spec, device=device) as model:
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
        with cache.use("xformer", device=device):
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
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        streamer: StreamedWeights = strategy._components[-1]
        original_register_hooks = streamer._register_hooks

        def broken_register_hooks(*args, **kwargs):
            original_register_hooks(*args, **kwargs)
            raise RuntimeError("simulated activate failure")

        monkeypatch.setattr(streamer, "_register_hooks", broken_register_hooks)

        with pytest.raises(RuntimeError, match="simulated activate failure"):
            strategy.activate("cuda")

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

            def activate(self, device: torch.device | str | None = None) -> None:
                del device
                events.append(f"activate:{self._name}")
                if self._raise:
                    # Simulate partial mutation before failing.
                    events.append(f"partial_state:{self._name}")
                    raise RuntimeError(f"{self._name} activate failed")

            def deactivate(self) -> None:
                events.append(f"deactivate:{self._name}")

        m = _make_block_model()
        strat = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strat._components = [_Recorder("A"), _Recorder("B"), _Recorder("C", raise_on_activate=True)]

        with pytest.raises(RuntimeError, match="C activate failed"):
            strat.activate("cpu")

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
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        strategy.activate("cuda")
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
    the activation device permanently, defeating ModelCache eviction."""

    @CUDA
    def test_constructed_has_no_params_on_activation_device(self) -> None:
        m = _make_block_model(num_blocks=4, width=8)
        target = torch.device("cuda")
        strategy = ModelOffloader(
            m, layers_attr="transformer_blocks", blocks_to_swap=2,
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
            m,
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
            m, layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            assert m.rope.table.is_pinned()
            strategy.activate("cuda")
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
                m,
                layers_attr="transformer_blocks", blocks_to_swap=1,
                stream_trainable_weights=True,
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
                m,
                layers_attr="transformer_blocks", blocks_to_swap=1,
                stream_trainable_weights=True,
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
                m,
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
                m,
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
                m,
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
                m,
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
                m,
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
            m,
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
            m,
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
            m,
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
            m,
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
            m,
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
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
        )
        try:
            strategy.activate("cuda")
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
        mover = TrainableWeights(m)
        assert mover.cache_bytes == 0
        mover.deactivate()

    def test_activate_and_deactivate_noop_when_no_trainable(self) -> None:
        m = _make_block_model()  # all frozen
        mover = TrainableWeights(m)
        try:
            mover.activate("cpu")
            mover.deactivate()
        finally:
            mover.deactivate()

    @CUDA
    def test_moves_trainable_param_to_device_on_activate(self) -> None:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora = nn.Parameter(torch.randn(4))  # trainable

        m = M()
        target = torch.device("cuda")
        mover = TrainableWeights(m)
        try:
            mover.activate(target)
            assert m.lora.is_cuda
            mover.deactivate()
            assert not m.lora.is_cuda  # back to CPU
        finally:
            mover.deactivate()

    def test_deactivate_idempotent(self) -> None:
        m = _make_block_model()
        mover = TrainableWeights(m)
        mover.deactivate()
        mover.deactivate()

    def test_skip_slots_walk_observes_late_requires_grad_changes(self) -> None:
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.streamed = nn.Linear(4, 4, bias=False)
                self.out_of_block = nn.Linear(4, 4, bias=False)

        m = M()
        for p in m.parameters():
            p.requires_grad = False

        from torch_offload.slots import iter_param_slots

        skip_slots = {
            s.slot for s in iter_param_slots(m.streamed)
        }
        mover = TrainableWeights(m, skip_slots=skip_slots)

        # Flip requires_grad after construction. The filtered mover
        # should preserve TrainableWeights' historical dynamic behavior
        # and pick it up at transition time.
        m.out_of_block.weight.requires_grad = True
        assert list(mover._iter_trainable_params()) == [m.out_of_block.weight]


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
    """StreamedWeights now handles in-block trainables natively via
    ``.data`` swap (preserves user Parameter identity for autograd /
    optimizer state); skip_slots is the surgical-exclude knob, not a
    trainable-required filter. The composer still routes out-of-block
    trainables to ``TrainableWeights`` via skip_slots."""

    def test_direct_unskipped_trainable_constructs(self) -> None:
        # Build a trainable block with no skip_slots. The streamer
        # accepts the trainable slot (handled via ``.data`` swap inside
        # ``_BlockPinnedStore``). The slot appears in the streamer's
        # slot_filter just like a frozen slot would.
        from torch_offload.slots import iter_param_slots

        block_0 = nn.Linear(4, 4, bias=False)  # default requires_grad=True
        block_1 = nn.Linear(4, 4, bias=False)
        trainable_slots = {
            s.slot for s in iter_param_slots(block_0) if s.param.requires_grad
        } | {
            s.slot for s in iter_param_slots(block_1) if s.param.requires_grad
        }
        streamer = StreamedWeights(
            blocks=[block_0, block_1],
            blocks_to_swap=1,
        )
        assert trainable_slots.issubset(streamer.slot_filter)
        # The user's Parameter object survives pinning — .data has been
        # repointed at the pinned clone, but the wrapper is unchanged
        # so optimizer state attached to it would still apply.
        assert isinstance(block_0.weight, nn.Parameter)
        assert block_0.weight.requires_grad

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
                m,
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
                m,
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )

    def test_all_trainable_same_parameter_cross_region_tie_raises(self) -> None:
        # The SAME trainable Parameter object aliased into two streamed
        # blocks is rejected: each block builds its own pinned clone,
        # so the per-block .data swap and the gather/scatter at step
        # time would diverge across the two clones — silent corruption
        # of the optimizer state. Block 1's layout matches block 0
        # (same Parameter shape, same slot names) so the layout check
        # passes; the tie check fires next.
        shared = nn.Parameter(torch.randn(4, 4), requires_grad=True)

        class TiedTrainableBlock(nn.Module):
            def __init__(self):
                super().__init__()
                # ``a`` references the cross-region-shared Parameter.
                self.a = shared

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [TiedTrainableBlock(), TiedTrainableBlock()]
                )

        m = M()

        with pytest.raises(
            ValueError, match="spans streamed regions",
        ):
            ModelOffloader(
                m,
                layers_attr="transformer_blocks", blocks_to_swap=1,
                stream_trainable_weights=True,
            )

    def test_all_trainable_same_parameter_default_mode_constructs(self) -> None:
        # Legacy/default mode skips all trainables in StreamedWeights and
        # moves the single shared Parameter through TrainableWeights, so
        # same-Parameter all-trainable cross-region ties remain valid.
        shared = nn.Parameter(torch.randn(4, 4), requires_grad=True)

        class TiedTrainableBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = shared

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [TiedTrainableBlock(), TiedTrainableBlock()]
                )

        m = M()
        strategy = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        assert m.transformer_blocks[0]._parameters["a"] is shared
        assert m.transformer_blocks[1]._parameters["a"] is shared
        strategy.deactivate()

    def test_all_trainable_same_parameter_intra_block_only_tie(self) -> None:
        # Pure intra-block aliasing (no cross-region): two slots inside
        # each block share the same trainable Parameter, but block 0's
        # shared Parameter is distinct from block 1's. ``_BlockPinnedStore``
        # dedups by ``id(param)`` per block, and the .data swap reaches
        # the shared Parameter so every aliased slot sees the update.
        # Layout signatures match because both blocks dedup to a single
        # entry (``a``) of identical shape/dtype.
        shared_0 = nn.Parameter(torch.randn(4, 4), requires_grad=True)
        shared_1 = nn.Parameter(torch.randn(4, 4), requires_grad=True)

        class TiedTrainableBlock(nn.Module):
            def __init__(self, p):
                super().__init__()
                self.a = p
                self.b = p

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [TiedTrainableBlock(shared_0), TiedTrainableBlock(shared_1)]
                )
                # HF-style flag so the in-block-trainable activate-time
                # guard would pass; we don't actually activate.
                for blk in self.transformer_blocks:
                    blk.gradient_checkpointing = True

        m = M()

        strategy = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
            stream_trainable_weights=True,
        )

        # Both aliased slots in block 0 still reference the same Parameter.
        assert m.transformer_blocks[0]._parameters["a"] is shared_0
        assert m.transformer_blocks[0]._parameters["b"] is shared_0
        del strategy

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
                m,
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
                m,
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

    def test_composer_routes_in_block_lora_to_streamer(self) -> None:
        # In the .data-only redesign, in-block trainables (LoRA adapters
        # nested inside streamed blocks) are managed by the streamer
        # itself via ``.data`` swap — no longer routed to
        # ``TrainableWeights``. The streamer's slot_filter therefore
        # covers BOTH frozen base slots AND trainable lora_a/lora_b.
        class M(nn.Module):
            def __init__(self, blocks):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(blocks)

        m = M([self._make_lora_block() for _ in range(2)])
        strat = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
            stream_trainable_weights=True,
        )
        try:
            # Each StreamedWeights's slot_filter contains BOTH frozen
            # base.weight and trainable lora_a/lora_b — the streamer
            # handles them uniformly.
            streamers = [
                c for c in strat._components if isinstance(c, StreamedWeights)
            ]
            assert len(streamers) == 1
            streamer = streamers[0]
            from torch_offload.slots import iter_param_slots
            for s in iter_param_slots(m):
                if "transformer_blocks" in s.name:
                    assert s.slot in streamer.slot_filter, (
                        f"in-block slot {s.name} (trainable={s.param.requires_grad}) "
                        f"missing from streamer's slot_filter"
                    )
                else:
                    assert s.slot not in streamer.slot_filter, (
                        f"out-of-block slot {s.name} leaked into streamer's "
                        f"slot_filter"
                    )
        finally:
            strat.deactivate()

    def test_default_routes_in_block_lora_to_trainable_weights(self) -> None:
        # Default mode preserves the historical contract: in-block
        # trainables are skipped by StreamedWeights and kept GPU-resident
        # by TrainableWeights while active.
        class M(nn.Module):
            def __init__(self, blocks):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(blocks)

        m = M([self._make_lora_block() for _ in range(2)])
        strat = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        try:
            streamers = [
                c for c in strat._components if isinstance(c, StreamedWeights)
            ]
            assert len(streamers) == 1
            streamer = streamers[0]
            from torch_offload.slots import iter_param_slots
            for s in iter_param_slots(m):
                if "transformer_blocks" in s.name and s.param.requires_grad:
                    assert s.slot not in streamer.slot_filter
                elif "transformer_blocks" in s.name:
                    assert s.slot in streamer.slot_filter
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
            m,
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


# ---------------------------------------------------------------------------
# Phase 1: training through streamed blocks under activation checkpointing
# ---------------------------------------------------------------------------
#
# The block streamer's slot pool reuses GPU storage across blocks via
# in-place ``copy_``, which bumps the slot tensor's autograd version
# counter on every load. Without checkpointing, the original forward's
# saved-tensor references into a slot are invalidated as soon as that
# slot is reused later in the same forward, and ``loss.backward()``
# raises ``RuntimeError: ... has been modified by an inplace
# operation`` before producing any grad.
#
# Activation checkpointing fixes this by deferring autograd-graph
# construction for each block to backward time (the recompute), at
# which point the forward-pre hook ensures the right block is loaded
# and the saved-tensor lifetimes don't span slot reuses.
#
# These tests pin Phase 1's contract: forward+backward through a
# streamed model produces baseline-matching grads under checkpointing,
# raises a loud, recognisable error without it, and the activate-time
# warning fires exactly when training-shape use is detected without
# HF-style ``gradient_checkpointing`` flags.

_OFFLOADER_LOGGER = "torch_offload.model_offloader"


class TestTrainingWithCheckpointing:
    @CUDA
    def test_grads_match_baseline_under_checkpointing(self) -> None:
        """Backward through a streamed model with per-block checkpointing
        produces grads that match a non-streamed CUDA baseline."""
        torch.manual_seed(42)
        m_baseline = _make_trainable_block_model(num_blocks=4, width=8)
        m_streamed = _make_trainable_block_model(num_blocks=4, width=8)
        m_streamed.load_state_dict(m_baseline.state_dict())

        x = torch.randn(2, 8, device="cuda")
        target = torch.randn(2, 8, device="cuda")

        m_baseline.to("cuda")
        out_b = m_baseline(x, use_checkpoint=True)
        ((out_b - target) ** 2).mean().backward()
        baseline_grads = {
            n: p.grad.detach().clone()
            for n, p in m_baseline.named_parameters()
            if p.grad is not None
        }
        assert baseline_grads, "baseline run produced no grads — bad fixture"

        # blocks_to_swap=2 + prefetch_count=0 → pool size 2 < 4 blocks,
        # so forward forces real slot reuse on blocks 2 and 3. That
        # reuse is what the checkpointing contract has to survive.
        offloader = ModelOffloader(
            m_streamed,
            layers_attr="transformer_blocks",
            blocks_to_swap=2, prefetch_count=0,
            stream_trainable_weights=True,
        )
        try:
            with offloader.use("cuda") as gpu_model:
                out_s = gpu_model(x, use_checkpoint=True)
                ((out_s - target) ** 2).mean().backward()
                torch.cuda.synchronize()
                streamed_grads = {
                    n: p.grad.detach().clone()
                    for n, p in gpu_model.named_parameters()
                    if p.grad is not None
                }
        finally:
            offloader.deactivate()

        assert set(baseline_grads) == set(streamed_grads), (
            f"grad keys differ: baseline={sorted(baseline_grads)}, "
            f"streamed={sorted(streamed_grads)}"
        )
        for name, g_baseline in baseline_grads.items():
            torch.testing.assert_close(
                streamed_grads[name], g_baseline, atol=1e-5, rtol=1e-5,
            )

    @CUDA
    def test_backward_without_checkpointing_raises_in_place_error(self) -> None:
        """Without checkpointing, slot reuse during forward bumps the
        version counter on a slot tensor whose forward-time reference
        autograd needs at backward. PyTorch's saved-tensor check
        catches this and raises."""
        torch.manual_seed(42)
        m = _make_trainable_block_model(num_blocks=4, width=8)
        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks",
            blocks_to_swap=2, prefetch_count=0,
        )
        try:
            with offloader.use("cuda") as gpu_model:
                x = torch.randn(2, 8, device="cuda")
                out = gpu_model(x, use_checkpoint=False)
                with pytest.raises(
                    RuntimeError, match="modified by an inplace operation",
                ):
                    out.sum().backward()
        finally:
            offloader.deactivate()


def _make_offloader_for_warning_test(model: nn.Module) -> ModelOffloader:
    """Build a CPU-targeted offloader for warning-logic unit tests.

    The warning helper is called directly on the constructed offloader
    rather than via ``activate()`` because checkpointing warnings are
    emitted only for CUDA streaming activation. The CUDA training tests
    above exercise the actual activation-site wiring; these tests pin
    the helper's *behaviour*, not its invocation site.
    """
    return ModelOffloader(
        model,
        layers_attr="transformer_blocks", blocks_to_swap=2,
    )


class TestTrainingWarning:
    """The activate-time warning is heuristic: it fires when
    ``model.training=True`` and at least one trainable param exists
    but no HuggingFace ``gradient_checkpointing`` flag is detected
    on the streamed blocks. Manual call-site checkpointing isn't
    visible from the module tree, so users wrapping that way will
    see a false-positive warning the first time they activate."""

    def test_fires_in_train_mode_with_trainables_and_no_flag(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        m = _make_trainable_block_model(num_blocks=4)
        offloader = _make_offloader_for_warning_test(m)
        with caplog.at_level(logging.WARNING, logger=_OFFLOADER_LOGGER):
            offloader._warn_if_training_without_checkpointing()
        warnings = [
            r for r in caplog.records
            if "no gradient_checkpointing flag" in r.message
        ]
        assert len(warnings) == 1, (
            f"expected one missing-checkpointing warning, got "
            f"{[r.message for r in caplog.records]}"
        )

    def test_silent_when_not_in_train_mode(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        m = _make_trainable_block_model(num_blocks=4)
        m.train(False)
        offloader = _make_offloader_for_warning_test(m)
        with caplog.at_level(logging.WARNING, logger=_OFFLOADER_LOGGER):
            offloader._warn_if_training_without_checkpointing()
        assert not any(
            "gradient_checkpointing" in r.message for r in caplog.records
        )

    def test_silent_when_no_trainable_params(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Inference user with model.training=True but nothing to train.
        # Heuristic should suppress the warning.
        m = _make_block_model(num_blocks=4)  # all frozen
        assert m.training, "fixture invariant: default modules are in train mode"
        offloader = _make_offloader_for_warning_test(m)
        with caplog.at_level(logging.WARNING, logger=_OFFLOADER_LOGGER):
            offloader._warn_if_training_without_checkpointing()
        assert not any(
            "gradient_checkpointing" in r.message for r in caplog.records
        )

    def test_silent_when_hf_flag_set_on_all_blocks(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        m = _make_trainable_block_model(num_blocks=4)
        for block in m.transformer_blocks:
            block.gradient_checkpointing = True
        offloader = _make_offloader_for_warning_test(m)
        with caplog.at_level(logging.WARNING, logger=_OFFLOADER_LOGGER):
            offloader._warn_if_training_without_checkpointing()
        assert not any(
            "gradient_checkpointing" in r.message for r in caplog.records
        )

    def test_silent_when_custom_predicate_accepts_blocks(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        m = _make_trainable_block_model(num_blocks=4)
        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
            is_block_checkpointed=lambda block: True,
        )
        with caplog.at_level(logging.WARNING, logger=_OFFLOADER_LOGGER):
            offloader._warn_if_training_without_checkpointing()
        assert not any(
            "gradient_checkpointing" in r.message for r in caplog.records
        )

    def test_skip_checkpointing_check_suppresses_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        m = _make_trainable_block_model(num_blocks=4)
        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=2,
            skip_checkpointing_check=True,
        )
        with caplog.at_level(logging.WARNING, logger=_OFFLOADER_LOGGER):
            offloader._warn_if_training_without_checkpointing()
        assert not any(
            "gradient_checkpointing" in r.message for r in caplog.records
        )

    def test_inconsistent_flags_emit_dedicated_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        m = _make_trainable_block_model(num_blocks=4)
        m.transformer_blocks[0].gradient_checkpointing = True
        # blocks 1..3 left unflagged
        offloader = _make_offloader_for_warning_test(m)
        with caplog.at_level(logging.WARNING, logger=_OFFLOADER_LOGGER):
            offloader._warn_if_training_without_checkpointing()
        assert any(
            "inconsistent gradient_checkpointing" in r.message
            for r in caplog.records
        ), [r.message for r in caplog.records]

    def test_fires_only_once_across_repeated_invocations(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        m = _make_trainable_block_model(num_blocks=4)
        offloader = _make_offloader_for_warning_test(m)
        with caplog.at_level(logging.WARNING, logger=_OFFLOADER_LOGGER):
            offloader._warn_if_training_without_checkpointing()
            offloader._warn_if_training_without_checkpointing()
            offloader._warn_if_training_without_checkpointing()
        n = sum(
            1 for r in caplog.records
            if "gradient_checkpointing" in r.message
        )
        assert n == 1, f"expected one warning across 3 invocations, got {n}"


# ---------------------------------------------------------------------------
# In-block trainable streaming (.data-only design)
# ---------------------------------------------------------------------------


def _make_lora_in_block_model(
    num_blocks: int = 4, width: int = 8, rank: int = 2,
) -> nn.Module:
    """LoRA-in-block model: each streamed block contains a frozen base
    Linear plus trainable lora_a / lora_b adapters. Forward computes
    ``base(x) + lora_b(lora_a(x))``. Optional per-block call-site
    checkpointing.
    """

    class LoRABlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = nn.Linear(width, width, bias=False)
            self.base.weight.requires_grad = False
            self.lora_a = nn.Linear(width, rank, bias=False)
            self.lora_b = nn.Linear(rank, width, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.base(x) + self.lora_b(self.lora_a(x))

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer_blocks = nn.ModuleList(
                LoRABlock() for _ in range(num_blocks)
            )

        def forward(
            self, x: torch.Tensor, *, use_checkpoint: bool = False
        ) -> torch.Tensor:
            for block in self.transformer_blocks:
                if use_checkpoint:
                    x = torch.utils.checkpoint.checkpoint(
                        block, x, use_reentrant=False,
                    )
                else:
                    x = block(x)
            return x

    return M()


class TestInBlockTrainableCheckpointingGuard:
    """``_enforce_checkpointing_for_trainable_streaming`` is a hard
    guard that prevents silent gradient corruption when in-block
    trainables are streamed without activation checkpointing. The
    guard is heuristic (HF-style ``gradient_checkpointing`` flag), so
    ``skip_checkpointing_check=True`` provides an escape hatch for
    call-site checkpointing.
    """

    def test_raises_without_checkpointing_flag(self) -> None:
        m = _make_lora_in_block_model(num_blocks=2)
        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
            stream_trainable_weights=True,
        )
        with pytest.raises(RuntimeError, match="gradient_checkpointing"):
            offloader._enforce_checkpointing_for_trainable_streaming()

    def test_passes_with_flag_on_every_block(self) -> None:
        m = _make_lora_in_block_model(num_blocks=2)
        for block in m.transformer_blocks:
            block.gradient_checkpointing = True
        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
            stream_trainable_weights=True,
        )
        offloader._enforce_checkpointing_for_trainable_streaming()  # no raise

    def test_raises_with_flag_on_only_some_blocks(self) -> None:
        m = _make_lora_in_block_model(num_blocks=4)
        m.transformer_blocks[0].gradient_checkpointing = True
        m.transformer_blocks[1].gradient_checkpointing = True
        # blocks 2, 3 unflagged
        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
            stream_trainable_weights=True,
        )
        with pytest.raises(RuntimeError, match="gradient_checkpointing"):
            offloader._enforce_checkpointing_for_trainable_streaming()

    def test_skip_checkpointing_check_suppresses_guard(self) -> None:
        m = _make_lora_in_block_model(num_blocks=2)
        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
            skip_checkpointing_check=True,
            stream_trainable_weights=True,
        )
        offloader._enforce_checkpointing_for_trainable_streaming()  # no raise

    def test_eval_mode_suppresses_guard(self) -> None:
        # Streamed trainable weights are safe for inference/eval activation:
        # the silent-corruption risk requires grad-enabled training through
        # streamed blocks.
        m = _make_lora_in_block_model(num_blocks=2)
        m.eval()
        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
            stream_trainable_weights=True,
        )
        offloader._enforce_checkpointing_for_trainable_streaming()  # no raise

    def test_silent_for_frozen_only_streamed_blocks(self) -> None:
        # Frozen-only blocks → no in-block trainables → no hard guard.
        # The frozen failure mode is autograd's loud version-counter
        # error, so a soft warning suffices (covered by
        # ``TestTrainingWarning``).
        m = _make_trainable_block_model(num_blocks=2)
        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        offloader._enforce_checkpointing_for_trainable_streaming()  # no raise

    def test_uncheckpointed_frozen_group_does_not_hard_fail(self) -> None:
        # Only streamers that actually manage trainable params need the
        # hard guard. Frozen-only groups still get the soft warning path
        # because their failure mode is autograd's loud version check.
        class FrozenBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(8, 8, bias=False)
                self.proj.weight.requires_grad = False

        m = _make_lora_in_block_model(num_blocks=2)
        m.frozen_blocks = nn.ModuleList(FrozenBlock() for _ in range(2))
        for block in m.transformer_blocks:
            block.gradient_checkpointing = True

        offloader = ModelOffloader(
            m,
            layers_attr=["transformer_blocks", "frozen_blocks"],
            blocks_to_swap=1,
            stream_trainable_weights=True,
        )
        offloader._enforce_checkpointing_for_trainable_streaming()  # no raise


class TestStreamedWeightsActivateTwice:
    @CUDA
    def test_double_activate_raises(self) -> None:
        m = _make_block_model(num_blocks=4)
        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        try:
            offloader.activate("cuda")
            # Reach into the streamer and call activate again — the
            # composer's activate handles its own teardown ExitStack,
            # but the streamer itself must hard-guard against
            # double-install of forward-pre hooks.
            streamer = offloader._streamers[0]
            with pytest.raises(RuntimeError, match="already.*active"):
                streamer.activate("cuda")
        finally:
            offloader.deactivate()


class TestInBlockTrainableStreamingEndToEnd:
    """Full forward+backward+optimizer.step cycle for an in-block-LoRA
    model. The .data-only design promises:

    - Backward through streamed blocks under checkpointing matches a
      non-streamed baseline (grads identical, no version-counter trip).
    - ``param.grad`` lives on GPU through backward via native
      ``AccumulateGrad`` (no custom hooks).
    - ``gather_for_step`` brings ``.data`` to GPU around the step;
      after exit, ``.data`` is back on pinned CPU.
    - User's ``Parameter`` object identity is preserved across the
      whole cycle so optimizer state is correct.
    """

    @CUDA
    def test_grads_match_baseline_with_in_block_trainables(self) -> None:
        torch.manual_seed(0)
        m_baseline = _make_lora_in_block_model(num_blocks=4, width=8, rank=2)
        m_streamed = _make_lora_in_block_model(num_blocks=4, width=8, rank=2)
        m_streamed.load_state_dict(m_baseline.state_dict())

        x = torch.randn(2, 8, device="cuda")
        target = torch.randn(2, 8, device="cuda")

        m_baseline.to("cuda")
        out_b = m_baseline(x, use_checkpoint=True)
        ((out_b - target) ** 2).mean().backward()
        baseline_grads = {
            n: p.grad.detach().clone()
            for n, p in m_baseline.named_parameters()
            if p.grad is not None
        }
        # LoRA params should have gradients on the baseline.
        assert any("lora_" in n for n in baseline_grads), (
            "fixture invariant: baseline produces lora_* gradients"
        )

        # Flag the blocks so the activate-time hard guard passes.
        for block in m_streamed.transformer_blocks:
            block.gradient_checkpointing = True

        offloader = ModelOffloader(
            m_streamed,
            layers_attr="transformer_blocks",
            blocks_to_swap=2, prefetch_count=0,
            stream_trainable_weights=True,
        )
        try:
            with offloader.use("cuda") as gpu_model:
                out_s = gpu_model(x, use_checkpoint=True)
                ((out_s - target) ** 2).mean().backward()
                torch.cuda.synchronize()
                streamed_grads = {
                    n: p.grad.detach().clone()
                    for n, p in gpu_model.named_parameters()
                    if p.grad is not None
                }
        finally:
            offloader.deactivate()

        assert set(baseline_grads) == set(streamed_grads), (
            f"grad keys differ: baseline={sorted(baseline_grads)}, "
            f"streamed={sorted(streamed_grads)}"
        )
        for name, g_baseline in baseline_grads.items():
            torch.testing.assert_close(
                streamed_grads[name], g_baseline, atol=1e-5, rtol=1e-5,
            )

    @CUDA
    def test_optimizer_step_updates_match_baseline(self) -> None:
        # Run one full step (forward + backward + step) on both a
        # baseline and a streamed model; verify resulting trainable
        # ``.data`` matches.
        torch.manual_seed(0)
        m_baseline = _make_lora_in_block_model(num_blocks=4, width=8, rank=2)
        m_streamed = _make_lora_in_block_model(num_blocks=4, width=8, rank=2)
        m_streamed.load_state_dict(m_baseline.state_dict())

        for block in m_streamed.transformer_blocks:
            block.gradient_checkpointing = True

        opt_baseline = torch.optim.SGD(
            [p for p in m_baseline.parameters() if p.requires_grad], lr=0.1,
        )
        opt_streamed = torch.optim.SGD(
            [p for p in m_streamed.parameters() if p.requires_grad], lr=0.1,
        )

        x = torch.randn(2, 8, device="cuda")
        target = torch.randn(2, 8, device="cuda")

        m_baseline.to("cuda")
        out_b = m_baseline(x, use_checkpoint=True)
        ((out_b - target) ** 2).mean().backward()
        opt_baseline.step()
        baseline_after = {
            n: p.detach().clone().cpu()
            for n, p in m_baseline.named_parameters()
            if p.requires_grad
        }

        offloader = ModelOffloader(
            m_streamed,
            layers_attr="transformer_blocks",
            blocks_to_swap=2, prefetch_count=0,
            stream_trainable_weights=True,
        )
        try:
            with offloader.use("cuda") as gpu_model:
                out_s = gpu_model(x, use_checkpoint=True)
                ((out_s - target) ** 2).mean().backward()
                with offloader.optimizer_step():
                    opt_streamed.step()
                opt_streamed.zero_grad()
        finally:
            offloader.deactivate()

        # After deactivate, all params live on pinned CPU. Compare the
        # source-of-truth (pinned host clones).
        streamed_after = {
            n: p.detach().clone().cpu()
            for n, p in m_streamed.named_parameters()
            if p.requires_grad
        }

        assert set(baseline_after) == set(streamed_after)
        for name, baseline_t in baseline_after.items():
            torch.testing.assert_close(
                streamed_after[name], baseline_t, atol=1e-5, rtol=1e-5,
            )

    @CUDA
    def test_gather_for_step_preserves_parameter_identity(self) -> None:
        # The user's Parameter object must survive the whole cycle —
        # otherwise optimizer state attached to it would be orphaned.
        torch.manual_seed(0)
        m = _make_lora_in_block_model(num_blocks=2, width=8, rank=2)
        for block in m.transformer_blocks:
            block.gradient_checkpointing = True

        # Snapshot Parameter ids BEFORE the offloader is constructed.
        initial_ids: dict[str, int] = {
            n: id(p) for n, p in m.named_parameters() if p.requires_grad
        }

        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks",
            blocks_to_swap=1, prefetch_count=0,
            stream_trainable_weights=True,
        )
        try:
            with offloader.use("cuda") as gpu_model:
                # Run forward to warm the slot pool.
                x = torch.randn(2, 8, device="cuda")
                _ = gpu_model(x, use_checkpoint=True)

                # Inside gather_for_step: trainable .data should be on GPU.
                with offloader.gather_for_step():
                    inside_ids = {
                        n: id(p)
                        for n, p in gpu_model.named_parameters()
                        if p.requires_grad
                    }
                    inside_devices = {
                        n: p.data.device.type
                        for n, p in gpu_model.named_parameters()
                        if p.requires_grad
                    }

                # After gather_for_step exit: .data should be back on CPU.
                after_ids = {
                    n: id(p)
                    for n, p in gpu_model.named_parameters()
                    if p.requires_grad
                }
                after_devices = {
                    n: p.data.device.type
                    for n, p in gpu_model.named_parameters()
                    if p.requires_grad
                }
        finally:
            offloader.deactivate()

        assert initial_ids == inside_ids == after_ids, (
            "Parameter object identity must be preserved across the "
            "gather_for_step boundary"
        )
        for n, dev in inside_devices.items():
            assert dev == "cuda", (
                f"{n} .data should be on cuda inside gather_for_step, got {dev}"
            )
        for n, dev in after_devices.items():
            assert dev == "cpu", (
                f"{n} .data should be back on cpu after gather_for_step, "
                f"got {dev}"
            )

    def test_cpu_pass_through_trainable_step_preserves_updates(self) -> None:
        torch.manual_seed(0)
        m = _make_lora_in_block_model(num_blocks=2, width=8, rank=2)
        param_ids = {
            n: id(p) for n, p in m.named_parameters() if p.requires_grad
        }
        before = {
            n: p.detach().clone()
            for n, p in m.named_parameters()
            if p.requires_grad
        }
        optimizer = torch.optim.SGD(
            [p for p in m.parameters() if p.requires_grad], lr=0.1,
        )

        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks",
            blocks_to_swap=1,
            stream_trainable_weights=True,
        )
        try:
            with offloader.use("cpu") as cpu_model:
                x = torch.randn(2, 8)
                target = torch.randn(2, 8)
                out = cpu_model(x)
                ((out - target) ** 2).mean().backward()
                with offloader.optimizer_step():
                    optimizer.step()
                active_after = {
                    n: p.detach().clone()
                    for n, p in cpu_model.named_parameters()
                    if p.requires_grad
                }

            after_deactivate = {
                n: p.detach().clone()
                for n, p in m.named_parameters()
                if p.requires_grad
            }
            assert param_ids == {
                n: id(p) for n, p in m.named_parameters() if p.requires_grad
            }
            assert any(
                not torch.allclose(active_after[n], before[n])
                for n in active_after
            )
            for name, active_value in active_after.items():
                torch.testing.assert_close(after_deactivate[name], active_value)
            for p in m.parameters():
                assert p.device == torch.device("cpu")
                assert p.is_pinned()
        finally:
            offloader.deactivate()


# ---------------------------------------------------------------------------
# Composer-level invariants from the .data-only redesign
# ---------------------------------------------------------------------------


class TestBlockGroupsDisjoint:
    """``ModelOffloader`` rejects configurations where the same module
    slot is owned by more than one streamer region. Catches duplicate
    blocks, parent/child overlap across groups, and the same block
    listed twice in one group — all of which would have multiple
    streamers (or one streamer twice) racing on the same slot's
    Parameter / .data swap."""

    def test_same_block_in_two_groups_raises(self) -> None:
        shared = nn.Linear(4, 4, bias=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.group_a = nn.ModuleList(
                    [shared, nn.Linear(4, 4, bias=False)]
                )
                self.group_b = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False), shared]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="disjoint module slots"):
            ModelOffloader(
                m,
                layers_attr=["group_a", "group_b"],
                blocks_to_swap=1,
            )

    def test_parent_child_overlap_raises(self) -> None:
        # group_a has a parent block whose child is also referenced
        # directly by group_b. Different block ids; same SlotOwnership
        # for the child's slots in both regions.
        child = nn.Linear(4, 4, bias=False)

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.child = child
                self.extra = nn.Linear(4, 4, bias=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.group_a = nn.ModuleList(
                    [Parent(), Parent()]
                )
                self.group_b = nn.ModuleList(
                    [child, nn.Linear(4, 4, bias=False)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="disjoint module slots"):
            ModelOffloader(
                m,
                layers_attr=["group_a", "group_b"],
                blocks_to_swap=1,
            )

    def test_same_block_twice_in_one_group_raises(self) -> None:
        shared = nn.Linear(4, 4, bias=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [shared, shared, nn.Linear(4, 4, bias=False)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="disjoint module slots"):
            ModelOffloader(
                m,
                layers_attr="transformer_blocks", blocks_to_swap=1,
            )


class TestPluggableCheckpointingDetection:
    """``is_block_checkpointed`` callable lets callers wire framework-
    specific detection without resorting to ``skip_checkpointing_check=True``
    (which silences the guard entirely). The default predicate stays
    HF-per-block-flag — conservative on purpose."""

    def test_custom_predicate_accepted(self) -> None:
        # A custom predicate returning True for every block lets the
        # guard pass even without per-block flags set.
        m = _make_lora_in_block_model(num_blocks=2)
        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
            is_block_checkpointed=lambda block: True,
            stream_trainable_weights=True,
        )
        offloader._enforce_checkpointing_for_trainable_streaming()  # no raise

    def test_custom_predicate_can_reject(self) -> None:
        # Predicate returning False on the second block triggers the
        # hard guard with the standard message.
        m = _make_lora_in_block_model(num_blocks=2)
        seen: list[nn.Module] = []

        def predicate(block: nn.Module) -> bool:
            seen.append(block)
            return len(seen) == 1  # True for first, False for second

        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
            is_block_checkpointed=predicate,
            stream_trainable_weights=True,
        )
        with pytest.raises(RuntimeError, match="is_block_checkpointed"):
            offloader._enforce_checkpointing_for_trainable_streaming()

    def test_skip_checkpointing_check_overrides_predicate(self) -> None:
        # skip_checkpointing_check=True short-circuits before the predicate
        # is even consulted. Use a sentinel predicate that would raise
        # if called to verify it isn't.
        m = _make_lora_in_block_model(num_blocks=2)

        def fail_predicate(block: nn.Module) -> bool:
            raise AssertionError("should not be called")

        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
            skip_checkpointing_check=True,
            is_block_checkpointed=fail_predicate,
            stream_trainable_weights=True,
        )
        offloader._enforce_checkpointing_for_trainable_streaming()  # no raise


# ---------------------------------------------------------------------------
# Streamer-level invariants from C1 + C2 (drain-and-evict + transactional gather)
# ---------------------------------------------------------------------------


class TestRevisedDataOnlyDesign:
    """Tests for the revised .data-only redesign: source-of-truth
    eviction during gather (C1), grad moved to CPU on deactivate
    (C1), transactional gather with stream isolation (C2), reentrant
    gather rejection (C2). Tests use the LoRA-in-block fixture."""

    @CUDA
    def test_deactivate_moves_trainable_grads_to_cpu(self) -> None:
        # Without the streamer's grad-on-deactivate handling, in-block
        # LoRA grads would stay GPU-resident after deactivate (which
        # defeats the whole memory contract).
        torch.manual_seed(0)
        m = _make_lora_in_block_model(num_blocks=2, width=8, rank=2)
        for block in m.transformer_blocks:
            block.gradient_checkpointing = True

        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks",
            blocks_to_swap=1, prefetch_count=0,
            stream_trainable_weights=True,
        )
        with offloader.use("cuda") as gpu_model:
            x = torch.randn(2, 8, device="cuda")
            target = torch.randn(2, 8, device="cuda")
            out = gpu_model(x, use_checkpoint=True)
            ((out - target) ** 2).mean().backward()
            torch.cuda.synchronize()
            # Mid-cycle: grads live on GPU (native AccumulateGrad).
            for n, p in gpu_model.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None and p.grad.device.type == "cuda", (
                        f"{n}: expected GPU grad mid-cycle, got {p.grad}"
                    )

        # After deactivate: grads (and data) are on CPU.
        for n, p in m.named_parameters():
            if p.requires_grad:
                assert p.data.device.type == "cpu", (
                    f"{n}: .data expected on cpu after deactivate, got {p.data.device}"
                )
                assert p.grad is None or p.grad.device.type == "cpu", (
                    f"{n}: .grad expected None or on cpu after deactivate, "
                    f"got {p.grad.device if p.grad is not None else None}"
                )
                assert p.grad is not None, (
                    f"{n}: deactivate should preserve grad (just move it), "
                    f"not clear it"
                )

    @CUDA
    def test_reactivate_moves_existing_grads_to_gpu_for_accumulation(self) -> None:
        # Gradient accumulation can span offloader activation windows.
        # Deactivate moves streamed trainable grads to CPU; the next
        # activate must move them back before AccumulateGrad adds the
        # next GPU-produced gradient.
        torch.manual_seed(0)
        m_baseline = _make_lora_in_block_model(num_blocks=2, width=8, rank=2)
        m_streamed = _make_lora_in_block_model(num_blocks=2, width=8, rank=2)
        m_streamed.load_state_dict(m_baseline.state_dict())

        batches = [
            (torch.randn(2, 8, device="cuda"), torch.randn(2, 8, device="cuda")),
            (torch.randn(2, 8, device="cuda"), torch.randn(2, 8, device="cuda")),
        ]

        m_baseline.to("cuda")
        for x, target in batches:
            out = m_baseline(x, use_checkpoint=True)
            ((out - target) ** 2).mean().backward()
        baseline_grads = {
            n: p.grad.detach().clone()
            for n, p in m_baseline.named_parameters()
            if p.grad is not None
        }

        for block in m_streamed.transformer_blocks:
            block.gradient_checkpointing = True

        offloader = ModelOffloader(
            m_streamed,
            layers_attr="transformer_blocks",
            blocks_to_swap=1, prefetch_count=0,
            stream_trainable_weights=True,
        )

        with offloader.use("cuda") as gpu_model:
            x, target = batches[0]
            out = gpu_model(x, use_checkpoint=True)
            ((out - target) ** 2).mean().backward()

        # Deactivate returns existing grads to CPU.
        assert all(
            p.grad is not None and p.grad.device.type == "cpu"
            for p in m_streamed.parameters()
            if p.requires_grad
        )

        with offloader.use("cuda") as gpu_model:
            assert all(
                p.grad is not None and p.grad.device.type == "cuda"
                for p in gpu_model.parameters()
                if p.requires_grad
            )
            x, target = batches[1]
            out = gpu_model(x, use_checkpoint=True)
            ((out - target) ** 2).mean().backward()
            streamed_grads = {
                n: p.grad.detach().clone()
                for n, p in gpu_model.named_parameters()
                if p.grad is not None
            }

        for name, baseline_grad in baseline_grads.items():
            torch.testing.assert_close(
                streamed_grads[name], baseline_grad, atol=1e-5, rtol=1e-5,
            )

    @CUDA
    def test_reentrant_gather_for_step_rejected(self) -> None:
        m = _make_lora_in_block_model(num_blocks=2, width=8, rank=2)
        for block in m.transformer_blocks:
            block.gradient_checkpointing = True

        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks",
            blocks_to_swap=1, prefetch_count=0,
            stream_trainable_weights=True,
        )
        try:
            with offloader.use("cuda") as gpu_model:
                x = torch.randn(2, 8, device="cuda")
                _ = gpu_model(x, use_checkpoint=True)
                with offloader.optimizer_step():
                    with pytest.raises(RuntimeError, match="reentrant"):
                        with offloader.optimizer_step():
                            pass
        finally:
            offloader.deactivate()

    @CUDA
    def test_body_exception_inside_gather_preserves_partial_step(
        self,
    ) -> None:
        # Exception inside the gather body must NOT roll back to stale
        # pinned bytes — the optimizer may have partially mutated
        # params before raising. Scatter on body exit (clean OR
        # exception); the user's exception handler sees the actual
        # failure rather than a silent revert.
        torch.manual_seed(0)
        m = _make_lora_in_block_model(num_blocks=2, width=8, rank=2)
        for block in m.transformer_blocks:
            block.gradient_checkpointing = True

        # Snapshot pre-step trainable .data.
        pre_step = {
            n: p.data.detach().clone()
            for n, p in m.named_parameters()
            if p.requires_grad
        }

        offloader = ModelOffloader(
            m,
            layers_attr="transformer_blocks",
            blocks_to_swap=1, prefetch_count=0,
            stream_trainable_weights=True,
        )
        sentinel = RuntimeError("simulated optimizer.step failure")
        try:
            with offloader.use("cuda") as gpu_model:
                x = torch.randn(2, 8, device="cuda")
                target = torch.randn(2, 8, device="cuda")
                out = gpu_model(x, use_checkpoint=True)
                ((out - target) ** 2).mean().backward()
                with pytest.raises(RuntimeError, match="simulated"):
                    with offloader.optimizer_step():
                        # Mutate .data on GPU to simulate a partial
                        # optimizer step before raising.
                        for n, p in gpu_model.named_parameters():
                            if p.requires_grad:
                                p.data.add_(1.0)
                        raise sentinel
        finally:
            offloader.deactivate()

        # The +1 mutation must have made it back to pinned despite
        # the exception inside the gather body.
        for n, post in (
            (n, p.data.detach().clone())
            for n, p in m.named_parameters()
            if p.requires_grad
        ):
            torch.testing.assert_close(
                post.cpu(), pre_step[n].cpu() + 1.0,
                atol=1e-5, rtol=1e-5,
                msg=lambda m, n=n: f"{n}: partial step not preserved ({m})",
            )

    @CUDA
    def test_prefetch_with_in_block_trainables_grads_match_baseline(
        self,
    ) -> None:
        # Real-world scenario: rank-256 LoRA on a many-block model.
        # Use prefetch_count>0 AND blocks_to_swap>0 so streaming is
        # actually exercising the prefetch slot pool with trainables.
        # Verify grads match a non-streamed baseline.
        torch.manual_seed(0)
        m_baseline = _make_lora_in_block_model(num_blocks=6, width=8, rank=2)
        m_streamed = _make_lora_in_block_model(num_blocks=6, width=8, rank=2)
        m_streamed.load_state_dict(m_baseline.state_dict())

        x = torch.randn(2, 8, device="cuda")
        target = torch.randn(2, 8, device="cuda")

        m_baseline.to("cuda")
        out_b = m_baseline(x, use_checkpoint=True)
        ((out_b - target) ** 2).mean().backward()
        baseline_grads = {
            n: p.grad.detach().clone()
            for n, p in m_baseline.named_parameters()
            if p.grad is not None
        }

        for block in m_streamed.transformer_blocks:
            block.gradient_checkpointing = True

        offloader = ModelOffloader(
            m_streamed,
            layers_attr="transformer_blocks",
            blocks_to_swap=3, prefetch_count=2,
            stream_trainable_weights=True,
        )
        try:
            with offloader.use("cuda") as gpu_model:
                out_s = gpu_model(x, use_checkpoint=True)
                ((out_s - target) ** 2).mean().backward()
                torch.cuda.synchronize()
                streamed_grads = {
                    n: p.grad.detach().clone()
                    for n, p in gpu_model.named_parameters()
                    if p.grad is not None
                }
        finally:
            offloader.deactivate()

        for name, g_baseline in baseline_grads.items():
            torch.testing.assert_close(
                streamed_grads[name], g_baseline, atol=1e-5, rtol=1e-5,
            )

    @CUDA
    def test_multi_iteration_adam_state_matches_baseline(self) -> None:
        # Adam keeps running first/second moment estimates (exp_avg,
        # exp_avg_sq) keyed on Parameter identity. Multi-iteration
        # streaming must preserve Parameter identity across gather/
        # scatter so optimizer state matches a non-streamed baseline.
        torch.manual_seed(0)
        m_baseline = _make_lora_in_block_model(num_blocks=4, width=8, rank=2)
        m_streamed = _make_lora_in_block_model(num_blocks=4, width=8, rank=2)
        m_streamed.load_state_dict(m_baseline.state_dict())

        for block in m_streamed.transformer_blocks:
            block.gradient_checkpointing = True

        opt_baseline = torch.optim.Adam(
            [p for p in m_baseline.parameters() if p.requires_grad], lr=1e-3,
        )
        opt_streamed = torch.optim.Adam(
            [p for p in m_streamed.parameters() if p.requires_grad], lr=1e-3,
        )

        # Same batch each iteration so updates are deterministic.
        x = torch.randn(2, 8, device="cuda")
        target = torch.randn(2, 8, device="cuda")

        m_baseline.to("cuda")
        for _ in range(3):
            opt_baseline.zero_grad()
            out_b = m_baseline(x, use_checkpoint=True)
            ((out_b - target) ** 2).mean().backward()
            opt_baseline.step()
        baseline_after = {
            n: p.detach().clone().cpu()
            for n, p in m_baseline.named_parameters()
            if p.requires_grad
        }

        offloader = ModelOffloader(
            m_streamed,
            layers_attr="transformer_blocks",
            blocks_to_swap=2, prefetch_count=0,
            stream_trainable_weights=True,
        )
        try:
            with offloader.use("cuda") as gpu_model:
                for _ in range(3):
                    opt_streamed.zero_grad()
                    out_s = gpu_model(x, use_checkpoint=True)
                    ((out_s - target) ** 2).mean().backward()
                    with offloader.optimizer_step():
                        opt_streamed.step()
        finally:
            offloader.deactivate()

        streamed_after = {
            n: p.detach().clone().cpu()
            for n, p in m_streamed.named_parameters()
            if p.requires_grad
        }
        for name, baseline_t in baseline_after.items():
            torch.testing.assert_close(
                streamed_after[name], baseline_t, atol=1e-5, rtol=1e-5,
            )
