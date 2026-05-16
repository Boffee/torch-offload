"""Tests for ``torch_offload.pinned_weights.PinnedWeights``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torch_offload import ModelStrategy, PinnedWeights

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_simple_model() -> nn.Module:
    """Two-layer Linear, frozen, CPU."""
    m = nn.Sequential(nn.Linear(8, 16, bias=False), nn.Linear(16, 8, bias=False))
    for p in m.parameters():
        p.requires_grad = False
    return m


# ---------------------------------------------------------------------------
# ModelStrategy protocol conformance
# ---------------------------------------------------------------------------


class TestModelStrategyConformance:
    def test_isinstance_runtime_check(self) -> None:
        pw = PinnedWeights(_make_simple_model())
        try:
            assert isinstance(pw, ModelStrategy)
        finally:
            pw.deactivate()

    def test_has_lifecycle_methods(self) -> None:
        pw = PinnedWeights(_make_simple_model())
        try:
            assert callable(pw.activate)
            assert callable(pw.deactivate)
            assert isinstance(pw.cache_bytes, int)
            assert pw.cache_bytes > 0
        finally:
            pw.deactivate()


# ---------------------------------------------------------------------------
# Lifecycle: activate / deactivate
# ---------------------------------------------------------------------------


class TestLifecycle:
    @CUDA
    def test_activate_returns_model_on_gpu(self) -> None:
        m = _make_simple_model()
        pw = PinnedWeights(m)
        try:
            pw.activate("cuda")
            assert pw.model is m
            for p in m.parameters():
                assert p.is_cuda
            pw.deactivate()
            for p in m.parameters():
                assert not p.is_cuda
                assert p.is_pinned()
        finally:
            pw.deactivate()

    def test_context_manager_protocol(self) -> None:
        m = _make_simple_model()
        pw = PinnedWeights(m)
        try:
            pinned_params = list(m.parameters())
            with pw.use("cpu") as model:
                assert model is m
                assert list(m.parameters()) == pinned_params
                for p in m.parameters():
                    assert p.is_pinned()
            for p in m.parameters():
                assert p.is_pinned()
        finally:
            pw.deactivate()

    def test_deactivate_when_not_active_is_noop(self) -> None:
        pw = PinnedWeights(_make_simple_model())
        try:
            pw.deactivate()
            pw.deactivate()
        finally:
            pw.deactivate()

    def test_repeated_activate_deactivate_cycle(self) -> None:
        pw = PinnedWeights(_make_simple_model())
        try:
            for _ in range(3):
                with pw.use("cpu"):
                    pass
        finally:
            pw.deactivate()

    def test_activate_accepts_device_without_constructor_default(self) -> None:
        m = _make_simple_model()
        pw = PinnedWeights(m)
        try:
            pw.activate(device="cpu")
            for p in m.parameters():
                assert p.device == torch.device("cpu")
            pw.deactivate()
        finally:
            pw.deactivate()

    def test_activate_without_any_device_raises(self) -> None:
        pw = PinnedWeights(_make_simple_model())
        try:
            with pytest.raises(ValueError, match="requires a device"):
                pw.activate()
        finally:
            pw.deactivate()


# ---------------------------------------------------------------------------
# Cleanup: drop the strategy + model refs to free pinned
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_drop_strategy_and_model_frees_pinned(self) -> None:
        # The "drop refs to free pinned" contract. Strategies don't have
        # a destructive close(); pinned tensors live in module slots
        # and are freed when the caller drops the model reference (and
        # the strategy reference, which is the only other holder).
        import gc
        import weakref

        m = _make_simple_model()
        pw = PinnedWeights(m)
        first_parent, first_leaf = pw._slots[0][1][0]
        slot_param_ref = weakref.ref(first_parent._parameters[first_leaf])
        pw.deactivate()
        assert slot_param_ref() is not None  # still alive via model slot
        del m, pw, first_parent
        gc.collect()
        assert slot_param_ref() is None  # GC freed it


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_rejects_unskipped_trainable_param(self) -> None:
        # Direct use without a composer: a trainable slot must hit the
        # contract guard. Slot replacement would orphan the user's
        # Parameter and break optimizer state — fail loudly rather than
        # silently freeze.
        m = nn.Linear(4, 4)  # default requires_grad=True, no buffers
        with pytest.raises(ValueError, match="cannot manage trainable slot"):
            PinnedWeights(m)

    def test_rejects_empty_model(self) -> None:
        # Frozen but with no params or buffers — nothing to manage.
        class Empty(nn.Module):
            pass
        m = Empty()
        with pytest.raises(ValueError, match="at least one frozen parameter"):
            PinnedWeights(m)

    def test_accepts_buffer_only_module(self) -> None:
        # A module with only registered buffers (no frozen params) is a
        # legitimate target — common for things like RoPE position tables
        # or sinusoidal embeddings. PinnedWeights should pin the buffers
        # and behave as a no-op for params.
        class BufferOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.randn(8, 4))

        m = BufferOnly()
        pw = PinnedWeights(m)
        try:
            assert pw.cache_bytes == 8 * 4 * 4  # float32
            assert m.table.is_pinned()
        finally:
            pw.deactivate()


# ---------------------------------------------------------------------------
# Tied-weight dedup — the V1 model-agnostic claim
# ---------------------------------------------------------------------------


class TestTiedWeightDedup:
    def _make_tied_model(self) -> tuple[nn.Module, nn.Embedding, nn.Linear]:
        """Embed + linear head with tied weights (the standard HF
        tie_weights() pattern: same Parameter under two names)."""
        embed = nn.Embedding(32, 16)
        head = nn.Linear(16, 32, bias=False)
        head.weight = embed.weight  # same Parameter object
        m = nn.Module()
        m.embed = embed
        m.head = head
        for p in m.parameters():
            p.requires_grad = False
        return m, embed, head

    def _make_distinct_param_tied_model(self) -> tuple[nn.Module, nn.Parameter, nn.Parameter]:
        """Two distinct Parameter objects sharing the same storage —
        the rarer case that the original PinnedWeights silently broke
        (e.g. quanto wrappers around shared inner _data)."""
        shared = torch.randn(8, 16, dtype=torch.bfloat16)
        a = nn.Parameter(shared, requires_grad=False)
        b = nn.Parameter(shared, requires_grad=False)  # distinct Parameter, same storage
        assert a is not b
        assert a.data.data_ptr() == b.data.data_ptr()
        m = nn.Module()
        m.a = a
        m.b = b
        return m, a, b

    def test_same_parameter_under_two_names_dedupes(self) -> None:
        m, embed, head = self._make_tied_model()
        pw = PinnedWeights(m)
        try:
            # Exactly one unique pinned buffer for the tied weight.
            assert len(pw._slots) == 1
            buf, locs = pw._slots[0]
            leaves = {leaf for _, leaf in locs}
            assert leaves == {"weight"}  # both 'embed.weight' and 'head.weight'
            assert len(locs) == 2
            # After construction, both slots reference the same Parameter
            # (the buf's cpu_param), preserving tying at the strongest level.
            assert m.embed._parameters["weight"] is m.head._parameters["weight"]
            assert m.embed.weight is buf.cpu_param
        finally:
            pw.deactivate()

    def test_distinct_params_sharing_storage_dedupe(self) -> None:
        m, _, _ = self._make_distinct_param_tied_model()
        pw = PinnedWeights(m)
        try:
            assert len(pw._slots) == 1
            _, locs = pw._slots[0]
            assert {leaf for _, leaf in locs} == {"a", "b"}
            # Both module slots now reference the same Parameter object.
            assert m._parameters["a"] is m._parameters["b"]
        finally:
            pw.deactivate()

    def test_zero_sized_same_parameter_under_two_names_dedupes(self) -> None:
        p = nn.Parameter(torch.empty(0), requires_grad=False)
        m = nn.Module()
        m.a = p
        m.b = p

        pw = PinnedWeights(m)
        try:
            assert len(pw._slots) == 1
            assert m._parameters["a"] is m._parameters["b"]
        finally:
            pw.deactivate()

    def test_cache_bytes_counts_tied_once(self) -> None:
        m, _, _ = self._make_tied_model()
        pw = PinnedWeights(m)
        try:
            # 32 * 16 * 4 (float32 default) = 2048 bytes for one buffer.
            # If the dedup were broken this would double.
            assert pw.cache_bytes == 32 * 16 * 4
        finally:
            pw.deactivate()

    @CUDA
    def test_tied_params_share_gpu_storage_on_activate(self) -> None:
        m, embed, head = self._make_tied_model()
        pw = PinnedWeights(m)
        try:
            with pw.use("cuda"):
                assert embed.weight.is_cuda
                assert head.weight.is_cuda
                assert embed.weight.data.data_ptr() == head.weight.data.data_ptr()
                # Stronger: same Parameter object.
                assert m.embed._parameters["weight"] is m.head._parameters["weight"]
        finally:
            pw.deactivate()

    @CUDA
    def test_distinct_tied_params_share_gpu_storage_on_activate(self) -> None:
        m, a, b = self._make_distinct_param_tied_model()
        pw = PinnedWeights(m)
        try:
            with pw.use("cuda"):
                # Slot identity comparison; the local `a` / `b` refs are
                # the now-orphaned originals (replaced in module slots).
                assert m._parameters["a"].is_cuda
                assert m._parameters["b"].is_cuda
                assert m._parameters["a"].data.data_ptr() == m._parameters["b"].data.data_ptr()
                assert m._parameters["a"] is m._parameters["b"]
        finally:
            pw.deactivate()


# ---------------------------------------------------------------------------
# Edge cases caught in review
# ---------------------------------------------------------------------------


class TestSharedSubmoduleAlias:
    def test_same_module_under_two_attrs(self) -> None:
        # Valid PyTorch model: m.a is m.b is the same nn.Linear instance.
        # named_modules() default removes the duplicate, but
        # named_parameters(remove_duplicate=False) walks both — the parent
        # lookup must use remove_duplicate=False too or KeyError.
        shared = nn.Linear(4, 4, bias=False)
        for p in shared.parameters():
            p.requires_grad = False
        m = nn.Module()
        m.a = shared
        m.b = shared
        pw = PinnedWeights(m)
        try:
            # One pinned buffer (same Parameter via aliased module).
            assert len(pw._slots) == 1
            _, locs = pw._slots[0]
            # Slots deduped by (id(parent), leaf) — only one physical
            # location even though there are two attribute paths.
            assert len(locs) == 1
        finally:
            pw.deactivate()

    def test_aliased_buffer(self) -> None:
        # Two distinct submodules sharing the same buffer tensor. The
        # default named_buffers() walks only one path; using
        # remove_duplicate=False plus storage-key dedup ensures both
        # paths get pinned consistently.
        shared_buf = torch.randn(4)

        class Inner(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.register_buffer("buf", b)
                self.weight = nn.Parameter(torch.randn(2), requires_grad=False)

        m = nn.Module()
        m.a = Inner(shared_buf)
        m.b = Inner(shared_buf)
        pw = PinnedWeights(m)
        try:
            # One pinned buffer covers both alias paths.
            assert len(pw._buffer_slots) == 1
            pinned, locs = pw._buffer_slots[0]
            assert pinned.is_pinned()
            assert len(locs) == 2
            # Both module slots reference the SAME pinned tensor.
            assert m.a.buf is pinned
            assert m.b.buf is pinned
        finally:
            pw.deactivate()

    def test_zero_sized_aliased_buffer(self) -> None:
        shared_buf = torch.empty(0)

        class Inner(nn.Module):
            def __init__(self, b: torch.Tensor):
                super().__init__()
                self.register_buffer("buf", b)
                self.weight = nn.Parameter(torch.randn(2), requires_grad=False)

        m = nn.Module()
        m.a = Inner(shared_buf)
        m.b = Inner(shared_buf)
        pw = PinnedWeights(m)
        try:
            assert len(pw._buffer_slots) == 1
            pinned, _locs = pw._buffer_slots[0]
            assert m.a.buf is pinned
            assert m.b.buf is pinned
        finally:
            pw.deactivate()


class TestMixedTrainableFrozenTied:
    def test_raises_when_tied_group_has_mixed_grad(self) -> None:
        # Two distinct Parameter objects sharing storage, one trainable
        # and one frozen. The contract guard fires on the trainable
        # alias the iteration encounters first; the composer's
        # detect_streaming_region_ties is the upstream layer that
        # rejects this with a tied-storage-specific message before
        # any pinning runs.
        shared = torch.randn(8, dtype=torch.bfloat16)
        a = nn.Parameter(shared, requires_grad=True)
        b = nn.Parameter(shared, requires_grad=False)
        m = nn.Module()
        m.a = a
        m.b = b
        with pytest.raises(ValueError, match="cannot manage trainable slot"):
            PinnedWeights(m)


class TestZeroSizedParams:
    def test_zero_sized_params_do_not_collapse(self) -> None:
        # Empty tensors all share data_ptr()==0; they must not dedup
        # into one buffer.
        m = nn.Module()
        m.a = nn.Parameter(torch.empty(0), requires_grad=False)
        m.b = nn.Parameter(torch.empty(0), requires_grad=False)
        # Need at least one non-empty frozen param so the constructor doesn't
        # reject the model. The empties should each be their own slot.
        m.c = nn.Parameter(torch.randn(4), requires_grad=False)
        pw = PinnedWeights(m)
        try:
            # 3 slots: a, b, c — empties did not collapse.
            assert len(pw._slots) == 3
        finally:
            pw.deactivate()


# ---------------------------------------------------------------------------
# Quanto end-to-end — the blocker Codex caught
# ---------------------------------------------------------------------------


class TestQuanto:
    def _make_quanto_model(self) -> nn.Module:
        """Build a tiny module whose 'weight' Parameter is a quanto
        WeightQBytesTensor. Simulates a quanto-quantized Linear without
        actually invoking quanto's quantize() pipeline (which requires
        hooking into nn.Linear in a way that's overkill for the test).
        """
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16)
        qt = WeightQBytesTensor.create(quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None)

        m = nn.Module()
        m.weight = nn.Parameter(qt, requires_grad=False)
        return m

    def test_quanto_constructor_repoints_to_pinned(self) -> None:
        # The bug Codex found: the original PinnedWeights did
        # `p.data = buf.cpu_param.data` which is a no-op for quanto.
        # The fix: swap module._parameters[leaf] = buf.cpu_param.
        # Verify the model now references pinned _data storage.
        m = self._make_quanto_model()
        original_data_ptr = m.weight._data.data_ptr()
        pw = PinnedWeights(m)
        try:
            # Inner _data must now point at pinned storage, not the original.
            assert m.weight._data.data_ptr() != original_data_ptr
            assert m.weight._data.is_pinned()
            assert m.weight._scale.is_pinned()
        finally:
            pw.deactivate()

    @CUDA
    def test_quanto_activate_moves_inner_to_cuda(self) -> None:
        m = self._make_quanto_model()
        pw = PinnedWeights(m)
        try:
            with pw.use("cuda"):
                assert m.weight._data.is_cuda
                assert m.weight._scale.is_cuda
            # Back to pinned after deactivate
            assert m.weight._data.is_pinned()
            assert m.weight._scale.is_pinned()
        finally:
            pw.deactivate()

    def test_quanto_close_releases_internal_state(self) -> None:
        # deactivate is idempotent; the model.s quanto
        # parameter still references its pinned-CPU storage. Pinned
        # memory is freed when the caller drops the model reference.
        m = self._make_quanto_model()
        pw = PinnedWeights(m)
        pw.deactivate()
        # Quanto wrapper still on CPU after deactivate.
        assert m.weight._data.is_pinned()
        assert m.weight._scale.is_pinned()


# ---------------------------------------------------------------------------
# cache_bytes accounting
# ---------------------------------------------------------------------------


class TestCacheBytes:
    def test_cache_bytes_positive(self) -> None:
        pw = PinnedWeights(_make_simple_model())
        try:
            assert pw.cache_bytes > 0
        finally:
            pw.deactivate()

    def test_cache_bytes_includes_buffers_when_requested(self) -> None:
        m = nn.Sequential(nn.Linear(4, 4, bias=False), nn.LayerNorm(4))
        for p in m.parameters():
            p.requires_grad = False
        pw_with = PinnedWeights(m, include_buffers=True)
        with_bytes = pw_with.cache_bytes

        m2 = nn.Sequential(nn.Linear(4, 4, bias=False), nn.LayerNorm(4))
        for p in m2.parameters():
            p.requires_grad = False
        pw_without = PinnedWeights(m2, include_buffers=False)
        assert pw_without.cache_bytes <= with_bytes
