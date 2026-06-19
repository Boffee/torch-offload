"""Tests for the block-streaming machinery in ``torch_offload``.

Covers ``ModelOffloader`` (the public composite),
and ``StreamedComponent`` (the per-block-list primitive).

CUDA-only tests gate on availability. CPU activation is pass-through
over the host-backed pinned state.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Future

import pytest
import torch
import torch.utils.checkpoint
from torch import nn

from torch_offload import (
    ModelOffloader,
    ModelOffloaderStore,
    ModelStrategy,
    PinnedComponent,
    StreamedComponent,
    StreamedComponentStore,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_model_offloader(
    model: nn.Module,
    *,
    blocks_attr: str | Sequence[str] | None = None,
    num_resident_blocks: int | None = None,
    num_prefetch_blocks: int = 2,
    cyclic: bool = False,
    stream_trainable_weights: bool = False,
    skip_checkpointing_check: bool = False,
    is_block_checkpointed: Callable[[nn.Module], bool] | None = None,
) -> ModelOffloader:
    store = ModelOffloaderStore.from_module(
        model,
        blocks_attr=blocks_attr,
        num_resident_blocks=num_resident_blocks,
        num_prefetch_blocks=num_prefetch_blocks,
        cyclic=cyclic,
        stream_trainable_weights=stream_trainable_weights,
    )
    return store.bind(
        model,
        skip_checkpointing_check=skip_checkpointing_check,
        is_block_checkpointed=is_block_checkpointed,
    )


def _make_streamed_component(
    blocks: Sequence[nn.Module],
    *,
    num_resident_blocks: int,
    num_prefetch_blocks: int = 2,
    cyclic: bool = False,
    blocks_path: str = "blocks",
    stream_trainable_weights: bool = True,
) -> StreamedComponent:
    model = _make_block_list_model(blocks, blocks_path)
    store = StreamedComponentStore.from_module(
        model,
        blocks_path=blocks_path,
        num_resident_blocks=num_resident_blocks,
        num_prefetch_blocks=num_prefetch_blocks,
        cyclic=cyclic,
        stream_trainable_weights=stream_trainable_weights,
    )
    return store.bind(model)


def _make_block_list_model(
    blocks: Sequence[nn.Module],
    blocks_path: str,
) -> nn.Module:
    class BlockListModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            parent: nn.Module = self
            parts = blocks_path.split(".")
            for part in parts[:-1]:
                child = nn.Module()
                setattr(parent, part, child)
                parent = child
            setattr(parent, parts[-1], nn.ModuleList(blocks))

    return BlockListModel()


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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        try:
            assert isinstance(strategy, ModelStrategy)
        finally:
            strategy.deactivate()

    def test_has_lifecycle_methods(self) -> None:
        m = _make_block_model()
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        try:
            assert callable(strategy.activate)
            assert callable(strategy.deactivate)
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Constructor pins; cache_bytes is final immediately
# ---------------------------------------------------------------------------


class TestConstructorPins:
    def test_constructor_pins_blocks(self) -> None:
        m = _make_block_model()
        store = ModelOffloaderStore.from_module(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        strategy = store.bind(m)
        try:
            # Block weights are pinned via registry replacement.
            for block in m.transformer_blocks:
                assert block.weight.is_pinned()
            # Non-block (embed/head) also pinned via composed PinnedComponent.
            assert m.embed.weight.is_pinned()
            assert m.head.weight.is_pinned()
            assert store.cache_bytes > 0
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Lifecycle: constructor → activate → deactivate
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_active_device_property_tracks_lifecycle(self) -> None:
        m = _make_block_model()
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        try:
            assert strategy.active_device is None
            with strategy.use("cpu"):
                assert strategy.active_device == torch.device("cpu")
            assert strategy.active_device is None
        finally:
            strategy.deactivate()

    @CUDA
    def test_activate_returns_model(self) -> None:
        m = _make_block_model()
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        try:
            strategy.activate("cuda")
            assert strategy._active_device == expected
            assert strategy._streamed_components[0]._active_device == expected
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

        strategy = _make_model_offloader(
            m_off,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        try:
            pinned_block_params = [
                block.weight for block in m_off.transformer_blocks
            ]
            with strategy.use("cpu") as cpu_model:
                assert strategy._active_device == torch.device("cpu")
                assert all(
                    s._active_device == torch.device("cpu")
                    for s in strategy._streamed_components
                )
                assert all(s._executor is None for s in strategy._streamed_components)
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
        strategy = _make_model_offloader(
            m, blocks_attr="transformer_blocks", num_resident_blocks=2,
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
        strategy = _make_model_offloader(
            m, blocks_attr="transformer_blocks", num_resident_blocks=2,
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
        strategy = _make_model_offloader(
            m, blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        try:
            strategy.activate("cuda")
            strategy.deactivate()
            strategy.activate("cuda")
            strategy.deactivate()
        finally:
            strategy.deactivate()

    def test_deactivate_when_not_active_is_noop(self) -> None:
        m = _make_block_model()
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        try:
            strategy.activate("cpu")

            def fail_activate(device: torch.device | str | None = None) -> None:
                del device
                raise AssertionError("component activation should not run")

            assert strategy._pinned_component is not None
            monkeypatch.setattr(
                strategy._pinned_component,
                "activate",
                fail_activate,
            )
            with pytest.raises(RuntimeError, match=r"already.*active"):
                strategy.activate("cpu")
        finally:
            strategy.deactivate()


class TestStreamedComponentBackendActivation:
    def test_direct_cpu_activation_uses_host_backed_weights(self) -> None:
        torch.manual_seed(42)
        m_eager = _make_block_model(num_blocks=4, width=8)
        m = _make_block_model(num_blocks=4, width=8)
        m.load_state_dict(m_eager.state_dict())

        x = torch.randn(2, 8)
        with torch.no_grad():
            expected = m_eager(x)

        streamer = _make_streamed_component(
            blocks=list(m.transformer_blocks),
            num_resident_blocks=2,
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

        streamer = _make_streamed_component(
            blocks=list(blocks),
            num_resident_blocks=1,
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
        streamer = _make_streamed_component(
            blocks=list(m.transformer_blocks),
            num_resident_blocks=1,
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        strategy.activate("cuda")
        strategy.deactivate()
        # Model is back in CPU/pinned state — usable, just without the
        # strategy's GPU streaming.
        for p in m.parameters():
            assert not p.is_cuda

    def test_deactivate_is_idempotent(self) -> None:
        m = _make_block_model()
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        strategy.deactivate()
        strategy.deactivate()  # no error

    @CUDA
    def test_deactivate_consumes_teardown_stack(self) -> None:
        m = _make_block_model()
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        strategy.activate("cuda")
        assert strategy._teardown_stack is not None
        strategy.deactivate()
        assert strategy._teardown_stack is None

    @CUDA
    def test_drop_strategy_without_deactivate_does_not_cycle(self) -> None:
        # Regression: StreamedComponent's forward-pre-hook closure used to
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        strategy.activate("cuda")  # installs hooks; no deactivate
        streamer = strategy._streamed_components[0]
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
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
        self, streamer: StreamedComponent,
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=3,
            num_prefetch_blocks=2, cyclic=True,
        )
        streamer: StreamedComponent = strategy._streamed_components[0]

        with strategy.use("cuda"):
            recorded, _ = self._record_prefetches(streamer)
            x = torch.randn(2, 8, device="cuda")
            m(x)  # iteration 1
            torch.cuda.synchronize()
            recorded.clear()
            m(x)  # iteration 2
            torch.cuda.synchronize()

        # 4 blocks * 2 prefetches per hook = 8 entries.
        # Per-hook expected (cyclic, num_prefetch_blocks=2):
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=3,
            num_prefetch_blocks=2, cyclic=False,
        )
        streamer: StreamedComponent = strategy._streamed_components[0]

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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=3,
            num_prefetch_blocks=1, cyclic=True,
        )
        streamer: StreamedComponent = strategy._streamed_components[0]

        with strategy.use("cuda"):
            recorded, _ = self._record_prefetches(streamer)
            x = torch.randn(2, 8, device="cuda")
            for idx in (3, 2, 1, 0):
                streamer._blocks[idx](x)
            torch.cuda.synchronize()

        # Per-hook (cyclic, num_prefetch_blocks=1):
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
        strategy = _make_model_offloader(
            m_off,
            blocks_attr="transformer_blocks", num_resident_blocks=3,
            num_prefetch_blocks=2, cyclic=True,
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
            num_prefetch_blocks=1, cyclic=True,
        )
        streamer: StreamedComponent = strategy._streamed_components[0]

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
        strategy = _make_model_offloader(
            m_off,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
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
    @CUDA
    def test_num_resident_blocks_above_num_blocks_clamps(self) -> None:
        # Values above the block count clamp at activation, so one
        # config stays valid across models of different depths.
        m = _make_block_model(num_blocks=4)
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=8,
        )
        try:
            with strategy.use("cuda"):
                m(torch.randn(2, 8, device="cuda"))
        finally:
            strategy.deactivate()

    def test_num_resident_blocks_must_be_at_least_one(self) -> None:
        m = _make_block_model(num_blocks=4)
        with pytest.raises(ValueError, match="num_resident_blocks"):
            _make_model_offloader(
                m,
                blocks_attr="transformer_blocks", num_resident_blocks=0,
            )

    def test_none_num_resident_blocks_disables_streaming(self) -> None:
        m = _make_block_model(num_blocks=4)
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=None,
        )
        try:
            assert not strategy._streamed_components
        finally:
            strategy.deactivate()

    def test_num_resident_blocks_without_blocks_attr_raises(self) -> None:
        m = _make_block_model(num_blocks=4)
        with pytest.raises(ValueError, match="requires blocks_attr"):
            _make_model_offloader(m, num_resident_blocks=2)

    def test_num_prefetch_blocks_must_be_non_negative(self) -> None:
        m = _make_block_model(num_blocks=4)
        with pytest.raises(ValueError, match="num_prefetch_blocks"):
            _make_model_offloader(
                m,
                blocks_attr="transformer_blocks", num_resident_blocks=2,
                num_prefetch_blocks=-1,
            )

    def test_empty_blocks_attr_raises(self) -> None:
        m = _make_block_model(num_blocks=4)
        with pytest.raises(ValueError, match="at least one path"):
            _make_model_offloader(
                m,
                blocks_attr=[], num_resident_blocks=2,
            )

    def test_blocks_attr_resolving_to_non_modulelist_raises(self) -> None:
        m = _make_block_model(num_blocks=4)
        with pytest.raises(TypeError, match="nn.ModuleList"):
            _make_model_offloader(
                m,
                blocks_attr="embed",  # an nn.Linear, not a ModuleList
                num_resident_blocks=2,
            )


# ---------------------------------------------------------------------------
# ModelCache integration: cache_bytes is final at construction
# ---------------------------------------------------------------------------


class TestModelCacheIntegration:
    def test_model_spec_default_reuses_canonical_model(self) -> None:
        from torch_offload import ModelCache, ModelSpec

        device = torch.device("cpu")
        factory_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return _make_block_model(num_blocks=4, width=8)

        cache = ModelCache(max_cache_bytes=10_000_000)
        spec = ModelSpec(
            key="xformer",
            estimated_cache_bytes=1024,
            factory=factory,
            blocks_attr="transformer_blocks",
            num_resident_blocks=2,
        )

        with cache.use(spec, device=device) as first_model:
            assert isinstance(first_model, nn.Module)
            assert all(not param.is_meta for param in first_model.parameters())
            assert all(not buffer.is_meta for buffer in first_model.buffers())
            x = torch.randn(2, 8)
            with torch.no_grad():
                _ = first_model(x)

        info = cache.info("xformer")
        assert info.cached
        assert info.cache_bytes is not None
        assert info.cache_bytes > 0
        assert info.active_count == 0

        # Second sequential use returns the same canonical model; factory
        # is not called again.
        with cache.use("xformer", device=device) as second_model:
            assert second_model is first_model
        assert factory_calls == 1

        cache.clear()

    def test_model_spec_default_rejects_nested_binding(self) -> None:
        from torch_offload import ModelCache, ModelInUseError, ModelSpec

        cache = ModelCache(max_cache_bytes=10_000_000)
        spec = ModelSpec(
            key="xformer",
            estimated_cache_bytes=1024,
            factory=lambda: _make_block_model(num_blocks=4, width=8),
            blocks_attr="transformer_blocks",
            num_resident_blocks=2,
        )

        with cache.use(spec, device="cpu"):
            with pytest.raises(ModelInUseError, match="xformer"):
                with cache.use("xformer", device="cpu"):
                    pass

        assert cache.info("xformer").active_count == 0
        cache.clear()

    def test_model_spec_skeleton_factory_supports_nested_bindings(self) -> None:
        from torch_offload import ModelCache, ModelSpec

        factory_calls = 0
        skeleton_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return _make_block_model(num_blocks=4, width=8)

        def skeleton_factory():
            nonlocal skeleton_calls
            skeleton_calls += 1
            with torch.device("meta"):
                return _make_block_model(num_blocks=4, width=8)

        cache = ModelCache(max_cache_bytes=10_000_000)
        spec = ModelSpec(
            key="xformer",
            estimated_cache_bytes=1024,
            factory=factory,
            skeleton_factory=skeleton_factory,
            blocks_attr="transformer_blocks",
            num_resident_blocks=2,
        )

        with cache.use(spec, device="cpu") as first_model:
            with cache.use("xformer", device="cpu") as second_model:
                assert first_model is not second_model
                assert all(not p.is_meta for p in first_model.parameters())
                assert all(not p.is_meta for p in second_model.parameters())
                assert cache.info("xformer").active_count == 2

        assert factory_calls == 1
        assert skeleton_calls == 2
        cache.clear()

    def test_model_spec_trainable_reuses_primary_model(self) -> None:
        from torch_offload import ModelCache, ModelSpec

        device = torch.device("cpu")
        factory_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return nn.Linear(8, 8, bias=False)

        cache = ModelCache(max_cache_bytes=10_000_000)
        spec = ModelSpec(
            key="trainable",
            estimated_cache_bytes=1024,
            factory=factory,
        )

        with cache.use(spec, device=device) as first_model:
            first_param_ids = [id(param) for param in first_model.parameters()]

        with cache.use("trainable", device=device) as second_model:
            assert second_model is first_model
            assert [id(param) for param in second_model.parameters()] == first_param_ids

        assert factory_calls == 1
        cache.clear()

    def test_model_spec_trainable_rejects_nested_binding(self) -> None:
        from torch_offload import ModelCache, ModelInUseError, ModelSpec

        cache = ModelCache(max_cache_bytes=10_000_000)
        spec = ModelSpec(
            key="trainable",
            estimated_cache_bytes=1024,
            factory=lambda: nn.Linear(8, 8, bias=False),
        )

        with cache.use(spec, device="cpu"):
            with pytest.raises(ModelInUseError, match="trainable"):
                with cache.use("trainable", device="cpu"):
                    pass

        assert cache.info("trainable").active_count == 0
        cache.clear()

    def test_model_spec_trainable_ignores_skeleton_factory(self) -> None:
        from torch_offload import ModelCache, ModelInUseError, ModelSpec

        skeleton_calls = 0

        def skeleton_factory():
            nonlocal skeleton_calls
            skeleton_calls += 1
            with torch.device("meta"):
                return nn.Linear(8, 8, bias=False)

        cache = ModelCache(max_cache_bytes=10_000_000)
        spec = ModelSpec(
            key="trainable",
            estimated_cache_bytes=1024,
            factory=lambda: nn.Linear(8, 8, bias=False),
            skeleton_factory=skeleton_factory,
        )

        with cache.use(spec, device="cpu") as first_model:
            with pytest.raises(ModelInUseError, match="trainable"):
                with cache.use("trainable", device="cpu"):
                    pass

        with cache.use("trainable", device="cpu") as second_model:
            assert second_model is first_model

        # skeleton_factory is ignored for trainable stores.
        assert skeleton_calls == 0
        cache.clear()


# ---------------------------------------------------------------------------
# Activate failure cleanup contract
# ---------------------------------------------------------------------------


class TestActivateFailureCleanup:
    @CUDA
    def test_partial_activate_failure_rolls_back_other_components(self, monkeypatch) -> None:
        # If a streamer's activate raises, the composite's `with stack:`
        # rolls back the already-activated components. _teardown_stack stays
        # None because pop_all()
        # was never reached. Caller's responsibility to drop the
        # strategy reference for full cleanup.
        m = _make_block_model()
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        streamer = strategy._streamed_components[0]
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
        strat = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        strat._pinned_component = None
        strat._streamed_components = [
            _Recorder("A"),
            _Recorder("B"),
            _Recorder("C", raise_on_activate=True),
        ]

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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        strategy.activate("cuda")
        streamer = strategy._streamed_components[0]
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
    footprint — the payoff of composer-owned stream/non-block partitioning.
    Without it, non-block siblings (embed, head, norms) would sit on
    the activation device permanently, defeating ModelCache eviction."""

    @CUDA
    def test_constructed_has_no_params_on_activation_device(self) -> None:
        m = _make_block_model(num_blocks=4, width=8)
        target = torch.device("cuda")
        strategy = _make_model_offloader(
            m, blocks_attr="transformer_blocks", num_resident_blocks=2,
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
        # No non-block PinnedComponent in components; store bytes come from blocks only.
        class BlockOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [nn.Linear(4, 4, bias=False) for _ in range(4)]
                )

        m = BlockOnly()
        for p in m.parameters():
            p.requires_grad = False
        store = ModelOffloaderStore.from_module(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        strategy = store.bind(m)
        try:
            # No PinnedComponent component, just StreamedComponent.
            assert strategy._pinned_component is None
            assert store.cache_bytes > 0  # block bytes only
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
        strategy = _make_model_offloader(
            m, blocks_attr="transformer_blocks", num_resident_blocks=2,
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
# Shared storage handled inside one component
# ---------------------------------------------------------------------------


class TestSharedStorageLocalBehavior:
    def test_extra_name_for_unstreamed_trainable_block_param_is_allowed(
        self,
    ) -> None:
        block = nn.Linear(4, 4, bias=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [block, nn.Linear(4, 4, bias=False)]
                )
                self.first_block = block

        m = M()

        strat = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
        )
        strat.deactivate()

    def test_intra_block_tied_params_are_preserved(self) -> None:
        shared_0 = torch.randn(8, 8)
        shared_1 = torch.randn(8, 8)

        class TiedBlock(nn.Module):
            def __init__(self, shared: torch.Tensor):
                super().__init__()
                self.attn_q = nn.Linear(8, 8, bias=False)
                self.attn_k = nn.Linear(8, 8, bias=False)
                self.attn_q.weight = nn.Parameter(shared, requires_grad=False)
                self.attn_k.weight = nn.Parameter(shared, requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [TiedBlock(shared_0), TiedBlock(shared_1)]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
        )
        try:
            for block in m.transformer_blocks:
                assert block.attn_q.weight is block.attn_k.weight
        finally:
            strategy.deactivate()

    def test_intra_block_tied_buffers_are_preserved(self) -> None:
        class TiedBufBlock(nn.Module):
            def __init__(self):
                super().__init__()
                shared = torch.randn(8)
                view_a = shared.view(8)
                view_b = shared.view(8)
                self.register_buffer("buf_a", view_a)
                self.register_buffer("buf_b", view_b)
                self.weight = nn.Parameter(torch.randn(2), requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [TiedBufBlock(), TiedBufBlock()]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
        )
        try:
            for block in m.transformer_blocks:
                assert block.buf_a is block.buf_b
                assert block.buf_a.is_pinned()
        finally:
            strategy.deactivate()

    def test_intra_block_same_buffer_object_is_preserved(self) -> None:
        class TiedBufBlock(nn.Module):
            def __init__(self):
                super().__init__()
                shared = torch.randn(8)
                self.register_buffer("buf_a", shared)
                self.register_buffer("buf_b", shared)
                self.weight = nn.Parameter(torch.randn(2), requires_grad=False)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [TiedBufBlock(), TiedBufBlock()]
                )

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
        )
        try:
            for block in m.transformer_blocks:
                assert block.buf_a is block.buf_b
                assert block.buf_a.is_pinned()
        finally:
            strategy.deactivate()

    def test_non_block_internal_tied_works(self) -> None:
        # Tied embed↔head WITHIN PinnedComponent: PinnedComponent handles
        # this via tensor-id dedup. Should not raise.
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        try:
            assert m.embed.weight is m.head.weight
            non_block = strategy._pinned_component
            assert non_block is not None
            assert non_block.param_names == {"embed.weight", "head.weight"}
            assert (
                len(
                    {
                        id(non_block._instance.params[name])
                        for name in non_block.param_names
                    }
                )
                == 1
            )
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Direct-parent state handling
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        try:
            assert m.scale_shift.is_pinned()
        finally:
            strategy.deactivate()

    def test_nested_blocks_attr_with_direct_root_param(self) -> None:
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="encoder.blocks", num_resident_blocks=2,
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        try:
            assert m.table.is_pinned()
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Block-internal buffers
# ---------------------------------------------------------------------------


class TestBlockBuffersPinned:
    @staticmethod
    def _make_tied_buffer_model(
        num_blocks: int = 2,
        *,
        device: torch.device | str = "cpu",
    ) -> nn.Module:
        class TiedBufferBlock(nn.Module):
            def __init__(self):
                super().__init__()
                shared = torch.randn(8, device=device)
                self.register_buffer("buf_a", shared.view(8))
                self.register_buffer("buf_b", shared.view(8))
                self.weight = nn.Parameter(
                    torch.randn(2, device=device),
                    requires_grad=False,
                )

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    TiedBufferBlock() for _ in range(num_blocks)
                )

        return M()

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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        try:
            for block in m.transformer_blocks:
                assert block.table.is_pinned(), (
                    "block buffer must be pinned for honest cache_bytes "
                    "and to avoid silently-synchronous H2D copies"
                )
        finally:
            strategy.deactivate()

    def test_constructor_does_not_call_module_to(self) -> None:
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2), requires_grad=False)
                self.register_buffer("table", torch.randn(2))

            def to(self, *args, **kwargs):
                raise AssertionError("constructor must pin directly")

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([Block(), Block()])

            def to(self, *args, **kwargs):
                raise AssertionError("constructor must pin directly")

        m = M()
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks",
            num_resident_blocks=1,
        )
        try:
            for block in m.transformer_blocks:
                assert block.weight.is_pinned()
                assert block.table.is_pinned()
        finally:
            strategy.deactivate()

    @CUDA
    def test_cuda_origin_tied_block_buffers_stay_tied(self) -> None:
        m = self._make_tied_buffer_model(device="cuda")
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks",
            num_resident_blocks=1,
        )
        try:
            for block in m.transformer_blocks:
                assert block.buf_a is block.buf_b
                assert block.buf_a.is_pinned()

            strategy.activate("cuda")
            resident = m.transformer_blocks[0]
            assert resident.buf_a is resident.buf_b
            assert resident.buf_a.is_cuda

            strategy.deactivate()
            for block in m.transformer_blocks:
                assert block.buf_a is block.buf_b
                assert block.buf_a.is_pinned()
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# StreamedComponent _activate_pool idempotency
# ---------------------------------------------------------------------------


class TestActivatePoolIdempotency:
    @CUDA
    def test_same_config_idempotent(self) -> None:
        m = _make_block_model()
        streamer = _make_streamed_component(
            list(m.transformer_blocks),
            num_resident_blocks=1,
        )
        streamer._activate_pool(2, torch.device("cuda"))
        pool_first = streamer._pool
        streamer._activate_pool(2, torch.device("cuda"))
        assert streamer._pool is pool_first

    @CUDA
    def test_mismatched_config_raises(self) -> None:
        m = _make_block_model()
        streamer = _make_streamed_component(
            list(m.transformer_blocks),
            num_resident_blocks=1,
        )
        streamer._activate_pool(2, torch.device("cuda"))
        with pytest.raises(ValueError, match="already activated"):
            streamer._activate_pool(3, torch.device("cuda"))
        with pytest.raises(ValueError, match="already activated"):
            streamer._activate_pool(2, torch.device("cpu"))


# ---------------------------------------------------------------------------
# Block layout check rejects heterogeneous block layouts
# ---------------------------------------------------------------------------


class TestBlockLayoutCompatibility:
    """Block 0 is the pool template; later blocks copy raw bytes into
    its target. ``Tensor.copy_`` silently casts dtype and silently
    broadcasts compatible shapes, so mismatches that don't trip the
    copy_ shape check would silently corrupt forward. The constructor's
    block layout check rejects them up front."""

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
            _make_streamed_component(
                blocks=list(m.blocks),
                num_resident_blocks=1,
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
            _make_streamed_component(
                blocks=[b0, b1],
                num_resident_blocks=1,
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
            _make_streamed_component(
                blocks=[A(), B()],
                num_resident_blocks=1,
            )

    def test_requires_grad_mismatch_raises(self) -> None:
        block_0 = nn.Linear(4, 4, bias=False)
        block_1 = nn.Linear(4, 4, bias=False)
        block_0.weight.requires_grad_(False)
        block_1.weight.requires_grad_(True)

        with pytest.raises(ValueError, match="layout differs"):
            _make_streamed_component(
                blocks=[block_0, block_1],
                num_resident_blocks=1,
            )

    def test_param_alias_topology_mismatch_raises(self) -> None:
        class TiedBlock(nn.Module):
            def __init__(self):
                super().__init__()
                shared = nn.Linear(4, 4, bias=False)
                shared.weight.requires_grad_(False)
                self.q = shared
                self.k = shared

        class UntiedBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = nn.Linear(4, 4, bias=False)
                self.k = nn.Linear(4, 4, bias=False)
                self.q.weight.requires_grad_(False)
                self.k.weight.requires_grad_(False)

        with pytest.raises(ValueError, match="layout differs"):
            _make_streamed_component(
                blocks=[TiedBlock(), UntiedBlock()],
                num_resident_blocks=1,
            )

    def test_buffer_shape_mismatch_raises(self) -> None:
        class Block(nn.Module):
            def __init__(self, buffer_size: int):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(4, 4), requires_grad=False,
                )
                self.register_buffer("table", torch.randn(buffer_size))

        with pytest.raises(ValueError, match="buffer layout differs"):
            _make_streamed_component(
                blocks=[Block(4), Block(8)],
                num_resident_blocks=1,
            )

    def test_buffer_stride_mismatch_raises(self) -> None:
        class Block(nn.Module):
            def __init__(self, table: torch.Tensor):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(4, 4), requires_grad=False,
                )
                self.register_buffer("table", table)

        contiguous = torch.randn(2, 3)
        non_contiguous = torch.randn(3, 2).t()
        assert contiguous.shape == non_contiguous.shape
        assert contiguous.stride() != non_contiguous.stride()

        with pytest.raises(ValueError, match="buffer layout differs"):
            _make_streamed_component(
                blocks=[Block(contiguous), Block(non_contiguous)],
                num_resident_blocks=1,
            )

    def test_buffer_name_mismatch_raises(self) -> None:
        class A(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(4, 4), requires_grad=False,
                )
                self.register_buffer("foo", torch.randn(4))

        class B(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(4, 4), requires_grad=False,
                )
                self.register_buffer("bar", torch.randn(4))

        with pytest.raises(ValueError, match="buffer layout differs"):
            _make_streamed_component(
                blocks=[A(), B()],
                num_resident_blocks=1,
            )

    def test_failure_leaves_model_unpinned_and_unmutated(self) -> None:
        # Strong-exception-safety: the validator runs in pass 1
        # (collect names) before pass 2 (pin) and pass 3 (restore
        # selected state). On a layout mismatch, the user's Parameter
        # objects must be the same identities and not pinned.
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
            _make_streamed_component(
                blocks=list(m.blocks),
                num_resident_blocks=1,
            )

        for block, orig_p, orig_pin in zip(
            m.blocks, original_params, original_pinned, strict=True,
        ):
            assert block.weight is orig_p, (
                "parameter identity changed despite pre-pin validation failure"
            )
            assert block.weight.is_pinned() == orig_pin, (
                "param was pinned despite pre-pin validation failure"
            )



# ---------------------------------------------------------------------------
# Multi-component cleanup ordering (ExitStack semantics)
# ---------------------------------------------------------------------------


class TestMultiComponentCleanup:
    @CUDA
    def test_pinned_deactivate_failure_still_runs_streamer_deactivate(
        self, monkeypatch,
    ) -> None:
        # ExitStack continues unwinding callbacks even when one raises.
        # If non-block PinnedComponent raises during deactivate, streamers
        # earlier in unwind order have still been deactivated.

        m = _make_block_model()
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
        )
        original_deactivate = PinnedComponent.deactivate

        def broken_deactivate(component: PinnedComponent) -> None:
            original_deactivate(component)
            raise RuntimeError("simulated pinned deactivate failure")

        monkeypatch.setattr(PinnedComponent, "deactivate", broken_deactivate)
        try:
            strategy.activate("cuda")
            assert m.embed.weight.is_cuda  # type: ignore[union-attr]

            with pytest.raises(RuntimeError, match="simulated pinned deactivate failure"):
                strategy.deactivate()

            # PinnedComponent restored registry entries before raising, and streamers
            # were already unwound in LIFO order.
            assert m.embed.weight.is_pinned()  # type: ignore[union-attr]
            assert not strategy._streamed_components[0]._hooks
            assert strategy._teardown_stack is None
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# StreamedComponent name selection
# ---------------------------------------------------------------------------


class TestStreamedNameSelection:
    def test_streamed_component_exposes_owned_block_local_names(self) -> None:
        m = _make_block_model()
        streamer = _make_streamed_component(
            blocks=list(m.transformer_blocks),
            num_resident_blocks=2,
        )
        try:
            assert streamer.streamed_param_names_by_block == [["weight"]] * len(
                m.transformer_blocks
            )
            assert streamer.streamed_buffer_names_by_block == [[]] * len(
                m.transformer_blocks
            )
        finally:
            streamer.deactivate()

    def test_streamed_component_exposes_addressable_param_names(self) -> None:
        m = _make_block_model()
        streamer = _make_streamed_component(
            blocks=list(m.transformer_blocks),
            num_resident_blocks=2,
            blocks_path="transformer_blocks",
        )
        try:
            assert streamer.param_names == {
                f"transformer_blocks.{i}.weight"
                for i in range(len(m.transformer_blocks))
            }
        finally:
            streamer.deactivate()

    def test_streamed_component_store_binds_compatible_model(self) -> None:
        torch.manual_seed(0)
        prototype = _make_block_model()
        target = _make_block_model()
        target_embed = target.embed.weight.detach().clone()

        store = StreamedComponentStore.from_module(
            prototype,
            blocks_path="transformer_blocks",
            num_resident_blocks=2,
        )
        streamer = store.bind(target)
        try:
            assert streamer.param_names == {
                f"transformer_blocks.{i}.weight"
                for i in range(len(target.transformer_blocks))
            }
            for prototype_block, target_block in zip(
                prototype.transformer_blocks,
                target.transformer_blocks,
                strict=True,
            ):
                assert target_block.weight.is_pinned()
                torch.testing.assert_close(target_block.weight, prototype_block.weight)
            torch.testing.assert_close(target.embed.weight, target_embed)
            assert not target.embed.weight.is_pinned()
        finally:
            streamer.deactivate()

    def test_model_offloader_store_binds_compatible_model(self) -> None:
        torch.manual_seed(0)
        prototype = _make_block_model()
        target = _make_block_model()

        store = ModelOffloaderStore.from_module(
            prototype,
            blocks_attr="transformer_blocks",
            num_resident_blocks=2,
        )
        strategy = store.bind(target)
        try:
            assert isinstance(strategy, ModelOffloader)
            assert strategy.model is target
            assert strategy.param_names == frozenset(
                name
                for name, _param in target.named_parameters(
                    remove_duplicate=False,
                )
            )
            assert strategy.buffer_names == frozenset()
            for name, param in target.named_parameters(remove_duplicate=False):
                prototype_param = dict(
                    prototype.named_parameters(remove_duplicate=False)
                )[name]
                assert param.is_pinned()
                torch.testing.assert_close(param, prototype_param)
        finally:
            strategy.deactivate()

    def test_model_offloader_exposes_all_managed_buffer_names(self) -> None:
        class BufferBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
                self.register_buffer("scale", torch.randn(2))

        class BufferModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.root_weight = nn.Parameter(
                    torch.randn(2, 2),
                    requires_grad=False,
                )
                self.register_buffer("root_scale", torch.randn(2))
                self.blocks = nn.ModuleList([BufferBlock(), BufferBlock()])

        model = BufferModel()
        strategy = _make_model_offloader(
            model,
            blocks_attr="blocks",
            num_resident_blocks=1,
        )
        try:
            assert strategy.param_names == frozenset(
                {
                    "root_weight",
                    "blocks.0.weight",
                    "blocks.1.weight",
                }
            )
            assert strategy.buffer_names == frozenset(
                {
                    "root_scale",
                    "blocks.0.scale",
                    "blocks.1.scale",
                }
            )
            assert strategy._streamed_components[0].buffer_names == frozenset(
                {
                    "blocks.0.scale",
                    "blocks.1.scale",
                }
            )
            instance, local_name = strategy._streamed_components[0]._resolve_buffer_name(
                "blocks.0.scale"
            )
            assert instance.module is model.blocks[0]
            assert local_name == "scale"
        finally:
            strategy.deactivate()

    def test_streamed_component_registers_post_copy_hook_by_param_name(self) -> None:
        m = _make_block_model()
        streamer = _make_streamed_component(
            blocks=list(m.transformer_blocks),
            num_resident_blocks=2,
            blocks_path="transformer_blocks",
        )
        try:
            handle = streamer.register_post_copy_hook(
                "transformer_blocks.1.weight",
                lambda _param: None,
            )
            key = streamer.post_copy_hook_key("transformer_blocks.1.weight")
            assert key in streamer._block_instances[1]._post_copy_hooks

            handle.remove()
            assert key not in streamer._block_instances[1]._post_copy_hooks
        finally:
            streamer.deactivate()

    def test_streamed_component_rejects_unknown_post_copy_hook_name(self) -> None:
        m = _make_block_model()
        streamer = _make_streamed_component(
            blocks=list(m.transformer_blocks),
            num_resident_blocks=2,
            blocks_path="transformer_blocks",
        )
        try:
            with pytest.raises(ValueError, match="not owned by this streamer"):
                streamer.register_post_copy_hook(
                    "transformer_blocks.10.weight",
                    lambda _param: None,
                )
        finally:
            streamer.deactivate()

    def test_model_offloader_partitions_streamed_and_pinned_names(
        self,
    ) -> None:
        m = _make_block_model()
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks",
            num_resident_blocks=2,
        )
        try:
            streamer = strategy._streamed_components[0]
            non_block = strategy._pinned_component
            assert non_block is not None

            assert streamer.streamed_param_names_by_block == [["weight"]] * len(
                m.transformer_blocks
            )
            assert non_block.param_names == {"embed.weight", "head.weight"}
            assert all(
                not name.startswith("transformer_blocks.")
                for name in non_block.param_names
            )
        finally:
            strategy.deactivate()


# ---------------------------------------------------------------------------
# Composer-driven trainable partitioning
# ---------------------------------------------------------------------------


class TestStreamedComponentContractGuard:
    """StreamedComponent now handles in-block trainables natively via
    ``.data`` swap (preserves user Parameter identity for autograd /
    optimizer state). The composer routes any non-streamed trainables to
    ``PinnedComponent``."""

    def test_direct_unskipped_trainable_constructs(self) -> None:
        block_0 = nn.Linear(4, 4, bias=False)  # default requires_grad=True
        block_1 = nn.Linear(4, 4, bias=False)
        streamer = _make_streamed_component(
            blocks=[block_0, block_1],
            num_resident_blocks=1,
        )
        assert streamer.streamed_param_names_by_block == [["weight"], ["weight"]]
        # The user's Parameter object survives pinning — .data has been
        # repointed at the pinned clone, but the wrapper is unchanged
        # so optimizer state attached to it would still apply.
        assert isinstance(block_0.weight, nn.Parameter)
        assert block_0.weight.requires_grad

class TestMixedGradTieDetection:
    """Mixed-grad shared storage is rejected by the owning component."""

    def test_intra_non_block_mixed_grad_tie_raises(self) -> None:
        # Two distinct Parameter objects sharing storage, both in the
        # PinnedComponent store, with mixed grad.
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
        with pytest.raises(ValueError, match="mixed requires_grad"):
            _make_model_offloader(
                m,
                blocks_attr="transformer_blocks", num_resident_blocks=1,
            )

    def test_all_trainable_distinct_parameter_tie_constructs(self) -> None:
        # Two distinct Parameter objects sharing storage, both trainable.
        # Default mode skips trainables in StreamedComponent, so PinnedComponent
        # owns and deduplicates the shared storage.
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
        )
        strategy.deactivate()

    @CUDA
    def test_all_trainable_distinct_parameter_tie_optimizer_step(
        self,
    ) -> None:
        # Exercise the exact storage-sharing shape allowed above through
        # CUDA activation and optimizer sync. The two Parameter wrappers
        # stay distinct for optimizer identity, but their data storage must
        # remain shared across activate/deactivate.
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
        m.eval()
        for p in m.transformer_blocks.parameters():
            p.requires_grad = False
        optimizer = torch.optim.SGD([a, b], lr=0.1)
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
        )
        try:
            assert m.alias_a.weight is a
            assert m.alias_b.weight is b
            assert a.data_ptr() == b.data_ptr()

            with strategy.use("cuda"):
                assert a.is_cuda
                assert b.is_cuda
                assert a.data_ptr() == b.data_ptr()
                a.grad = torch.ones_like(a)
                b.grad = torch.full_like(b, 2.0)
                with strategy.optimizer_step():
                    optimizer.step()
                    expected = a.detach().cpu().clone()
                    torch.testing.assert_close(b.detach().cpu(), expected)
                optimizer.zero_grad(set_to_none=True)

            assert a.is_pinned()
            assert b.is_pinned()
            assert a.data_ptr() == b.data_ptr()
            torch.testing.assert_close(a.detach(), expected)
            torch.testing.assert_close(b.detach(), expected)

            with strategy.use("cuda"):
                assert a.data_ptr() == b.data_ptr()
                torch.testing.assert_close(a.detach().cpu(), expected)
                torch.testing.assert_close(b.detach().cpu(), expected)
        finally:
            strategy.deactivate()

    def test_all_trainable_same_parameter_default_mode_constructs(self) -> None:
        # Default mode skips all trainables in StreamedComponent and manages
        # them through PinnedComponent, so all-trainable shared storage
        # remain valid.
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
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
        )
        assert m.transformer_blocks[0]._parameters["a"] is shared
        assert m.transformer_blocks[1]._parameters["a"] is shared
        strategy.deactivate()

    def test_all_trainable_same_parameter_intra_block_only_tie(self) -> None:
        # Pure intra-block aliasing: two names inside
        # each block share the same trainable Parameter, but block 0's
        # shared Parameter is distinct from block 1's. Streamed block
        # stores dedup by storage per block, and the .data swap reaches
        # the shared Parameter so every aliased name sees the update.
        # The pool layout matches because both blocks dedup to a single
        # storage group of identical shape/dtype.
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

        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
            stream_trainable_weights=True,
        )

        # Both aliased names in block 0 still reference the same Parameter.
        assert m.transformer_blocks[0]._parameters["a"] is shared_0
        assert m.transformer_blocks[0]._parameters["b"] is shared_0
        del strategy

    @CUDA
    def test_streamed_trainable_constructor_moves_existing_grads_to_cpu(self) -> None:
        class TrainableBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4, device="cuda"))
                self.weight.grad = torch.randn_like(self.weight)
                self.gradient_checkpointing = True

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    [TrainableBlock(), TrainableBlock()]
                )

        m = M()
        strategy = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks",
            num_resident_blocks=1,
            stream_trainable_weights=True,
        )
        try:
            for block in m.transformer_blocks:
                assert block.weight.device.type == "cpu"
                assert block.weight.is_pinned()
                assert block.weight.grad is not None
                assert block.weight.grad.device.type == "cpu"
        finally:
            strategy.deactivate()


class TestLoRAInBlockRouting:
    """LoRA-shaped models: blocks contain frozen base layers plus
    trainable adapter layers. The composer must route the base to
    StreamedComponent and, in default mode, the adapters to PinnedComponent; neither
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
        # itself via ``.data`` swap. The streamer therefore owns BOTH
        # frozen base.weight and trainable lora_a/lora_b names.
        class M(nn.Module):
            def __init__(self, blocks):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(blocks)

        m = M([self._make_lora_block() for _ in range(2)])
        strat = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
            stream_trainable_weights=True,
        )
        try:
            streamers = strat._streamed_components
            assert len(streamers) == 1
            streamer = streamers[0]
            assert streamer.streamed_param_names_by_block == [
                ["base.weight", "lora_a.weight", "lora_b.weight"],
                ["base.weight", "lora_a.weight", "lora_b.weight"],
            ]
        finally:
            strat.deactivate()

    def test_default_routes_in_block_lora_to_pinned_component(self) -> None:
        # Default mode skips in-block trainables in StreamedComponent and
        # keeps them GPU-resident through PinnedComponent while active.
        class M(nn.Module):
            def __init__(self, blocks):
                super().__init__()
                self.transformer_blocks = nn.ModuleList(blocks)

        m = M([self._make_lora_block() for _ in range(2)])
        strat = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
        )
        try:
            streamers = strat._streamed_components
            assert len(streamers) == 1
            streamer = streamers[0]
            assert streamer.streamed_param_names_by_block == [
                ["base.weight"],
                ["base.weight"],
            ]
            pinned = strat._pinned_component
            assert pinned is not None
            assert pinned.param_names == {
                "transformer_blocks.0.lora_a.weight",
                "transformer_blocks.0.lora_b.weight",
                "transformer_blocks.1.lora_a.weight",
                "transformer_blocks.1.lora_b.weight",
            }
        finally:
            strat.deactivate()

    def test_composer_partitions_names_correctly(self) -> None:
        # Through-test: the composer subtracts streamed full names from
        # model names to form PinnedComponent include-name sets. Verify that
        # streamed block names are excluded and non-streamed trainables are
        # included.
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

        strat = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
        )
        try:
            pinned = strat._pinned_component
            assert pinned is not None
            # PinnedComponent manages every non-streamed parameter, including
            # trainable_bias.
            assert pinned.param_names == {"frozen_head.weight", "trainable_bias"}
            for name, param in m.named_parameters(remove_duplicate=False):
                if name in {"frozen_head.weight", "trainable_bias"}:
                    assert name in pinned.param_names
                else:
                    assert name not in pinned.param_names, (
                        f"param {name} (requires_grad={param.requires_grad}) "
                        "leaked into PinnedComponent"
                    )
        finally:
            strat.deactivate()


# ---------------------------------------------------------------------------
# Phase 1: training through streamed blocks under activation checkpointing
# ---------------------------------------------------------------------------
#
# The block streamer's target pool reuses GPU storage across blocks via
# in-place ``copy_``, which bumps the target tensor's autograd version
# counter on every load. Without checkpointing, the original forward's
# saved-tensor references into a target are invalidated as soon as that
# target is reused later in the same forward, and ``loss.backward()``
# raises ``RuntimeError: ... has been modified by an inplace
# operation`` before producing any grad.
#
# Activation checkpointing fixes this by deferring autograd-graph
# construction for each block to backward time (the recompute), at
# which point the forward-pre hook ensures the right block is loaded
# and the saved-tensor lifetimes don't span target reuses.
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

        # num_resident_blocks=2 + num_prefetch_blocks=0 → pool size 2 < 4 blocks,
        # so forward forces real target reuse on blocks 2 and 3. That
        # reuse is what the checkpointing contract has to survive.
        offloader = _make_model_offloader(
            m_streamed,
            blocks_attr="transformer_blocks",
            num_resident_blocks=2, num_prefetch_blocks=0,
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
    def test_streamed_trainable_cpu_optimizer_step_without_context(self) -> None:
        """A plain ``optimizer.step()`` outside ``use()`` (and outside
        ``optimizer_step()``) runs the optimizer on CPU over the pinned
        trainable weights: states allocate on the host, and the in-place
        update writes through to what the next forward streams.

        This pins the context-free CPU-optimizer pattern — the offloader
        leaves the trainable ``.data`` (pinned CPU) and ``.grad`` (CPU) ready
        for a host-side step, so ``optimizer_step()`` is only needed when you
        want the step on GPU for speed. fp32 trainables make the in-place
        update a correct master-weight update.
        """
        torch.manual_seed(0)

        class Block(nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.lin = nn.Linear(width, width, bias=False)  # trainable
                self.gradient_checkpointing = True

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(self.lin(x))

        class M(nn.Module):
            def __init__(self, width: int, n: int) -> None:
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    Block(width) for _ in range(n)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.transformer_blocks:
                    x = torch.utils.checkpoint.checkpoint(
                        block, x, use_reentrant=False
                    )
                return x

        m = M(8, 3)
        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks",
            num_resident_blocks=1,
            num_prefetch_blocks=0,
            stream_trainable_weights=True,
        )
        opt = torch.optim.AdamW(m.parameters(), lr=0.1)
        weight = m.transformer_blocks[0].lin.weight
        try:
            for _ in range(2):
                x = torch.randn(4, 8, device="cuda")
                with offloader.use("cuda") as gpu_model:
                    gpu_model(x).sum().backward()

                # Deactivated: trainable data + grad are host-resident.
                assert weight.device.type == "cpu" and weight.is_pinned()
                assert weight.grad is not None
                assert weight.grad.device.type == "cpu"

                before = weight.detach().clone()
                opt.step()  # CPU step — no optimizer_step() context
                opt.zero_grad(set_to_none=True)
                assert not torch.equal(weight.detach(), before)
                updated = weight.detach().clone()

                # Write-through: the next forward streams the updated weight.
                with offloader.use("cuda"):
                    assert torch.equal(
                        m.transformer_blocks[0].lin.weight.detach().cpu(),
                        updated,
                    )

            # The optimizer's state (Adam moments) lives on the host, not GPU.
            assert all(
                not t.is_cuda
                for state in opt.state.values()
                for t in state.values()
                if torch.is_tensor(t)
            )
        finally:
            offloader.deactivate()

    @CUDA
    def test_backward_without_checkpointing_raises_in_place_error(self) -> None:
        """Without checkpointing, target reuse during forward bumps the
        version counter on a target tensor whose forward-time reference
        autograd needs at backward. PyTorch's saved-tensor check
        catches this and raises."""
        torch.manual_seed(42)
        m = _make_trainable_block_model(num_blocks=4, width=8)
        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks",
            num_resident_blocks=2, num_prefetch_blocks=0,
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
    return _make_model_offloader(
        model,
        blocks_attr="transformer_blocks", num_resident_blocks=2,
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

    def test_silent_when_default_mode_streams_no_block_state(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class AllTrainableBlocks(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer_blocks = nn.ModuleList(
                    nn.Linear(4, 4, bias=False) for _ in range(2)
                )

        m = AllTrainableBlocks()
        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks",
            num_resident_blocks=1,
        )
        try:
            assert not offloader._has_streamed_blocks
            assert offloader._streamed_components[0].param_names == frozenset()
            assert offloader._streamed_components[0].streamed_buffer_names_by_block == [
                [],
                [],
            ]

            with caplog.at_level(logging.WARNING, logger=_OFFLOADER_LOGGER):
                offloader._warn_if_training_without_checkpointing()

            assert not any(
                "gradient_checkpointing" in r.message for r in caplog.records
            )
        finally:
            offloader.deactivate()

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
        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
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
        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=2,
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
        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
            stream_trainable_weights=True,
        )
        with pytest.raises(RuntimeError, match="gradient_checkpointing"):
            offloader._enforce_checkpointing_for_trainable_streaming()

    def test_passes_with_flag_on_every_block(self) -> None:
        m = _make_lora_in_block_model(num_blocks=2)
        for block in m.transformer_blocks:
            block.gradient_checkpointing = True
        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
            stream_trainable_weights=True,
        )
        offloader._enforce_checkpointing_for_trainable_streaming()  # no raise

    def test_raises_with_flag_on_only_some_blocks(self) -> None:
        m = _make_lora_in_block_model(num_blocks=4)
        m.transformer_blocks[0].gradient_checkpointing = True
        m.transformer_blocks[1].gradient_checkpointing = True
        # blocks 2, 3 unflagged
        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
            stream_trainable_weights=True,
        )
        with pytest.raises(RuntimeError, match="gradient_checkpointing"):
            offloader._enforce_checkpointing_for_trainable_streaming()

    def test_skip_checkpointing_check_suppresses_guard(self) -> None:
        m = _make_lora_in_block_model(num_blocks=2)
        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
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
        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
            stream_trainable_weights=True,
        )
        offloader._enforce_checkpointing_for_trainable_streaming()  # no raise

    def test_silent_for_frozen_only_streamed_blocks(self) -> None:
        # Frozen-only blocks → no in-block trainables → no hard guard.
        # The frozen failure mode is autograd's loud version-counter
        # error, so a soft warning suffices (covered by
        # ``TestTrainingWarning``).
        m = _make_trainable_block_model(num_blocks=2)
        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
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

        offloader = _make_model_offloader(
            m,
            blocks_attr=["transformer_blocks", "frozen_blocks"],
            num_resident_blocks=1,
            stream_trainable_weights=True,
        )
        offloader._enforce_checkpointing_for_trainable_streaming()  # no raise


class TestStreamedComponentActivateTwice:
    @CUDA
    def test_double_activate_raises(self) -> None:
        m = _make_block_model(num_blocks=4)
        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=3,
        )
        try:
            offloader.activate("cuda")
            # Reach into the streamer and call activate again — the
            # composer's activate handles its own teardown ExitStack,
            # but the streamer itself must hard-guard against
            # double-install of forward-pre hooks.
            streamer = offloader._streamed_components[0]
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

        offloader = _make_model_offloader(
            m_streamed,
            blocks_attr="transformer_blocks",
            num_resident_blocks=2, num_prefetch_blocks=0,
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
    def test_default_optimizer_step_updates_match_baseline(self) -> None:
        # Default mode keeps in-block LoRA trainables out of the streamer;
        # ModelOffloader.optimizer_step must still sync their PinnedComponent
        # CPU cache after the optimizer mutates CUDA data.
        torch.manual_seed(0)
        m_baseline = _make_lora_in_block_model(num_blocks=4, width=8, rank=2)
        m_offloaded = _make_lora_in_block_model(num_blocks=4, width=8, rank=2)
        m_offloaded.load_state_dict(m_baseline.state_dict())

        for block in m_offloaded.transformer_blocks:
            block.gradient_checkpointing = True

        opt_baseline = torch.optim.SGD(
            [p for p in m_baseline.parameters() if p.requires_grad], lr=0.1,
        )
        opt_offloaded = torch.optim.SGD(
            [p for p in m_offloaded.parameters() if p.requires_grad], lr=0.1,
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

        offloader = _make_model_offloader(
            m_offloaded,
            blocks_attr="transformer_blocks",
            num_resident_blocks=2, num_prefetch_blocks=0,
        )
        try:
            with offloader.use("cuda") as gpu_model:
                out_s = gpu_model(x, use_checkpoint=True)
                ((out_s - target) ** 2).mean().backward()
                with offloader.optimizer_step():
                    opt_offloaded.step()
                opt_offloaded.zero_grad()
        finally:
            offloader.deactivate()

        offloaded_after = {
            n: p.detach().clone().cpu()
            for n, p in m_offloaded.named_parameters()
            if p.requires_grad
        }

        assert set(baseline_after) == set(offloaded_after)
        for name, baseline_t in baseline_after.items():
            torch.testing.assert_close(
                offloaded_after[name], baseline_t, atol=1e-5, rtol=1e-5,
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

        offloader = _make_model_offloader(
            m_streamed,
            blocks_attr="transformer_blocks",
            num_resident_blocks=2, num_prefetch_blocks=0,
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

        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks",
            num_resident_blocks=1, num_prefetch_blocks=0,
            stream_trainable_weights=True,
        )
        try:
            with offloader.use("cuda") as gpu_model:
                # Run forward to warm the target pool.
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

        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks",
            num_resident_blocks=1,
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


class TestPluggableCheckpointingDetection:
    """``is_block_checkpointed`` callable lets callers wire framework-
    specific detection without resorting to ``skip_checkpointing_check=True``
    (which silences the guard entirely). The default predicate stays
    HF-per-block-flag — conservative on purpose."""

    def test_custom_predicate_accepted(self) -> None:
        # A custom predicate returning True for every block lets the
        # guard pass even without per-block flags set.
        m = _make_lora_in_block_model(num_blocks=2)
        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
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

        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
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

        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks", num_resident_blocks=1,
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

        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks",
            num_resident_blocks=1, num_prefetch_blocks=0,
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

        offloader = _make_model_offloader(
            m_streamed,
            blocks_attr="transformer_blocks",
            num_resident_blocks=1, num_prefetch_blocks=0,
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

        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks",
            num_resident_blocks=1, num_prefetch_blocks=0,
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

        offloader = _make_model_offloader(
            m,
            blocks_attr="transformer_blocks",
            num_resident_blocks=1, num_prefetch_blocks=0,
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
        # Use num_prefetch_blocks>0 AND num_resident_blocks < num blocks
        # so streaming actually exercises the prefetch target pool
        # with trainables.
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

        offloader = _make_model_offloader(
            m_streamed,
            blocks_attr="transformer_blocks",
            num_resident_blocks=3, num_prefetch_blocks=2,
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

        offloader = _make_model_offloader(
            m_streamed,
            blocks_attr="transformer_blocks",
            num_resident_blocks=2, num_prefetch_blocks=0,
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
