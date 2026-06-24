"""Tests for ``torch_offload.model_offloader.ModelOffloader``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest
import torch
from torch import nn

from torch_offload import (
    ModelOffloader,
    ModelOffloaderStore,
    ModelStrategy,
    PinnedComponent,
    PinnedComponentStore,
)

from tests.conftest import pinned_component

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_model_offloader(
    model: nn.Module,
    *,
    blocks_attr: str | Sequence[str] | None = None,
    num_resident_blocks: int | None = None,
    num_prefetch_blocks: int = 2,
    cyclic: bool = False,
    stream_trainable_weights: bool = False,
) -> ModelOffloader:
    store = ModelOffloaderStore.from_module(
        model,
        blocks_attr=blocks_attr,
        num_resident_blocks=num_resident_blocks,
        num_prefetch_blocks=num_prefetch_blocks,
        cyclic=cyclic,
        stream_trainable_weights=stream_trainable_weights,
    )
    return store.bind(model)


def _make_simple_model() -> nn.Module:
    """Two-layer Linear, frozen, CPU."""
    m = nn.Sequential(nn.Linear(8, 16, bias=False), nn.Linear(16, 8, bias=False))
    for p in m.parameters():
        p.requires_grad = False
    return m


def _unique_pinned_param_count(pw: ModelOffloader) -> int:
    return len({id(pinned) for pinned in pinned_component(pw)._instance.params.values()})


def _unique_pinned_buffer_count(pw: ModelOffloader) -> int:
    return len({id(pinned) for pinned in pinned_component(pw)._instance.buffers.values()})


# ---------------------------------------------------------------------------
# ModelStrategy protocol conformance
# ---------------------------------------------------------------------------


class TestModelStrategyConformance:
    def test_isinstance_runtime_check(self) -> None:
        pw = _make_model_offloader(_make_simple_model())
        try:
            assert isinstance(pw, ModelStrategy)
        finally:
            pw.deactivate()

    def test_component_is_not_top_level_strategy(self) -> None:
        model = _make_simple_model()
        component = PinnedComponentStore.from_module(model).bind(model)
        try:
            assert not isinstance(component, ModelStrategy)
        finally:
            component.deactivate()

    def test_component_constructor_is_not_public_factory(self) -> None:
        model = _make_simple_model()
        with pytest.raises(TypeError, match="PinnedComponentStore.from_module"):
            PinnedComponent(cast(Any, model))

    def test_offloader_constructor_requires_bound_composite(self) -> None:
        with pytest.raises(TypeError, match="composite"):
            cast(Any, ModelOffloader)(_make_simple_model())

    def test_has_lifecycle_methods(self) -> None:
        pw = _make_model_offloader(_make_simple_model())
        try:
            assert callable(pw.activate)
            assert callable(pw.deactivate)
        finally:
            pw.deactivate()


class TestPinnedComponentStoreBind:
    def test_bind_allows_empty_noop_component(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        module.register_buffer("running", torch.randn(2))

        store = PinnedComponentStore.from_module(
            module,
            include_param_names=set(),
            include_buffer_names=set(),
        )

        assert store.cache_bytes == 0
        assert store.param_names == frozenset()
        assert store.buffer_names == frozenset()

        component = store.bind(module)
        try:
            assert component.param_names == frozenset()
            assert component.buffer_names == frozenset()
            component.activate("cpu")
        finally:
            component.deactivate()

    def test_binds_existing_store_to_multiple_components(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        prototype.register_buffer("running", torch.randn(2))
        store = PinnedComponentStore.from_module(prototype)

        first = store.bind(prototype)
        second_model = nn.Module()
        second_model.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        second_model.register_buffer("running", torch.randn(2))
        second = store.bind(second_model)

        try:
            assert store.param_names == frozenset({"weight"})
            assert first.param_names == store.param_names
            assert second.param_names == store.param_names
            assert store.buffer_names == frozenset({"running"})
            assert first.buffer_names == store.buffer_names
            assert second.buffer_names == store.buffer_names

            assert prototype.weight.data_ptr() == second_model.weight.data_ptr()
            assert prototype.weight is not second_model.weight
            assert prototype.running is second_model.running

            first.activate("cpu")
            second.activate("cpu")
        finally:
            first.deactivate()
            second.deactivate()

    def test_bind_propagates_module_mismatch_errors(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = PinnedComponentStore.from_module(prototype)

        target = nn.Module()
        target.weight = nn.Parameter(torch.randn(3, 2), requires_grad=False)

        with pytest.raises(ValueError, match="Param 'weight' layout mismatch"):
            store.bind(target)


# ---------------------------------------------------------------------------
# Trainable params
# ---------------------------------------------------------------------------


class TestTrainableParams:
    def test_accepts_trainable_param_and_preserves_identity_on_cpu(self) -> None:
        m = nn.Linear(4, 2, bias=False)
        param = m.weight
        opt = torch.optim.SGD(m.parameters(), lr=0.1)

        pw = _make_model_offloader(m)
        try:
            assert m.weight is param
            assert m.weight.requires_grad
            assert m.weight.is_pinned()

            before = m.weight.detach().clone()
            with pw.use("cpu"):
                loss = m(torch.ones(1, 4)).sum()
                loss.backward()
                with pw.optimizer_step():
                    opt.step()
                opt.zero_grad(set_to_none=True)

            assert m.weight is param
            assert m.weight.is_pinned()
            assert not torch.equal(m.weight.detach(), before)
        finally:
            pw.deactivate()

    def test_optimizer_step_rejects_reentrant_entry(self) -> None:
        pw = _make_model_offloader(nn.Linear(4, 2, bias=False))
        try:
            with pytest.raises(RuntimeError, match="reentrant"):
                with pw.optimizer_step():
                    with pw.optimizer_step():
                        pass
        finally:
            pw.deactivate()

    def test_cpu_active_optimizer_step_rejects_reentrant_entry(self) -> None:
        pw = _make_model_offloader(nn.Linear(4, 2, bias=False))
        try:
            with pw.use("cpu"):
                with pytest.raises(RuntimeError, match="reentrant"):
                    with pw.optimizer_step():
                        with pw.optimizer_step():
                            pass
        finally:
            pw.deactivate()

    @CUDA
    def test_cuda_trainable_identity_survives_activate_deactivate(self) -> None:
        m = nn.Linear(4, 2, bias=False)
        param = m.weight

        pw = _make_model_offloader(m)
        try:
            with pw.use("cuda"):
                assert m.weight is param
                assert m.weight.is_cuda
                assert m.weight.requires_grad
            assert m.weight is param
            assert m.weight.is_pinned()
        finally:
            pw.deactivate()

    @CUDA
    def test_cuda_optimizer_step_copies_updates_back_to_pinned(self) -> None:
        m = nn.Linear(4, 1, bias=False)
        param = m.weight
        opt = torch.optim.SGD(m.parameters(), lr=0.25)

        pw = _make_model_offloader(m)
        try:
            with pw.use("cuda"):
                loss = m(torch.ones(1, 4, device="cuda")).sum()
                loss.backward()
                with pw.optimizer_step():
                    opt.step()
                opt.zero_grad(set_to_none=True)
                updated = m.weight.detach().cpu().clone()

            assert m.weight is param
            assert m.weight.is_pinned()
            assert torch.equal(m.weight.detach(), updated)

            with pw.use("cuda"):
                assert torch.equal(m.weight.detach().cpu(), updated)
        finally:
            pw.deactivate()

    @CUDA
    def test_cuda_optimizer_step_copies_back_on_body_exception(self) -> None:
        m = nn.Linear(4, 1, bias=False)
        pw = _make_model_offloader(m)
        try:
            with pw.use("cuda"):
                with pytest.raises(RuntimeError, match="boom"):
                    with pw.optimizer_step():
                        m.weight.data.add_(1)
                        updated = m.weight.detach().cpu().clone()
                        raise RuntimeError("boom")

            assert torch.equal(m.weight.detach(), updated)
        finally:
            pw.deactivate()

    @CUDA
    def test_cuda_deactivate_moves_trainable_grad_to_cpu(self) -> None:
        # A pinned trainable's grad must follow its data to CPU on
        # deactivate, mirroring the streamed component. Otherwise grads
        # linger on the GPU (pinning device memory and stranding the
        # gradient off-host), which breaks the context-free CPU step.
        torch.manual_seed(0)
        m = nn.Linear(4, 2, bias=False)
        pw = _make_model_offloader(m)
        try:
            with pw.use("cuda"):
                m(torch.ones(1, 4, device="cuda")).sum().backward()
                torch.cuda.synchronize()
                # Mid-cycle: grad lives on GPU (native AccumulateGrad).
                assert m.weight.grad is not None
                assert m.weight.grad.device.type == "cuda"

            # Deactivated: data and grad are both host-resident, and the
            # grad is preserved (moved, not cleared).
            assert m.weight.data.device.type == "cpu" and m.weight.is_pinned()
            assert m.weight.grad is not None
            assert m.weight.grad.device.type == "cpu"
        finally:
            pw.deactivate()

    @CUDA
    def test_cuda_context_free_cpu_optimizer_step(self) -> None:
        """A plain ``optimizer.step()`` outside ``use()`` (and outside
        ``optimizer_step()``) runs the optimizer on CPU over the pinned
        trainable: Adam state allocates on the host and the in-place update
        writes through to what the next forward loads. Pins the context-free
        CPU-optimizer pattern for the pinned (whole-model) path; fp32
        trainables make the update a correct master-weight update."""
        torch.manual_seed(0)
        m = nn.Linear(8, 8, bias=False)
        pw = _make_model_offloader(m)
        opt = torch.optim.AdamW(m.parameters(), lr=0.1)
        try:
            for _ in range(2):
                x = torch.randn(4, 8, device="cuda")
                with pw.use("cuda") as gpu_model:
                    gpu_model(x).sum().backward()

                # Deactivated: trainable data + grad are host-resident.
                assert m.weight.device.type == "cpu" and m.weight.is_pinned()
                assert m.weight.grad is not None
                assert m.weight.grad.device.type == "cpu"

                before = m.weight.detach().clone()
                opt.step()  # CPU step — no optimizer_step() context
                opt.zero_grad(set_to_none=True)
                assert not torch.equal(m.weight.detach(), before)
                updated = m.weight.detach().clone()

                # Write-through: the next forward loads the updated weight.
                with pw.use("cuda"):
                    assert torch.equal(m.weight.detach().cpu(), updated)

            # The optimizer's state (Adam moments) lives on the host.
            assert all(
                not t.is_cuda
                for state in opt.state.values()
                for t in state.values()
                if torch.is_tensor(t)
            )
        finally:
            pw.deactivate()


# ---------------------------------------------------------------------------
# Lifecycle: activate / deactivate
# ---------------------------------------------------------------------------


class TestLifecycle:
    @CUDA
    def test_activate_returns_model_on_gpu(self) -> None:
        m = _make_simple_model()
        pw = _make_model_offloader(m)
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
        pw = _make_model_offloader(m)
        try:
            # CPU activate restores frozen params to fresh wrappers over the
            # same pinned storage (the materialized CPU wrapper is built on
            # demand, not cached), so compare stable pinned-storage pointers.
            pinned_ptrs = [p.data_ptr() for p in m.parameters()]
            with pw.use("cpu") as model:
                assert model is m
                assert [p.data_ptr() for p in m.parameters()] == pinned_ptrs
                for p in m.parameters():
                    assert p.is_pinned()
            for p in m.parameters():
                assert p.is_pinned()
        finally:
            pw.deactivate()

    def test_deactivate_when_not_active_is_noop(self) -> None:
        pw = _make_model_offloader(_make_simple_model())
        try:
            pw.deactivate()
            pw.deactivate()
        finally:
            pw.deactivate()

    def test_double_activate_raises(self) -> None:
        pw = _make_model_offloader(_make_simple_model())
        try:
            pw.activate("cpu")
            with pytest.raises(RuntimeError, match=r"already.*active"):
                pw.activate("cpu")
        finally:
            pw.deactivate()

    def test_repeated_activate_deactivate_cycle(self) -> None:
        pw = _make_model_offloader(_make_simple_model())
        try:
            for _ in range(3):
                with pw.use("cpu"):
                    pass
        finally:
            pw.deactivate()

    def test_activate_accepts_device_without_constructor_default(self) -> None:
        m = _make_simple_model()
        pw = _make_model_offloader(m)
        try:
            pw.activate(device="cpu")
            for p in m.parameters():
                assert p.device == torch.device("cpu")
            pw.deactivate()
        finally:
            pw.deactivate()

    def test_activate_without_any_device_raises(self) -> None:
        pw = _make_model_offloader(_make_simple_model())
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
        # a destructive close(); pinned tensors live in module state
        # and are freed when the caller drops the model reference (and
        # the strategy reference, which is the only other holder).
        import gc
        import weakref

        m = _make_simple_model()
        pw = _make_model_offloader(m)
        module_param_ref = weakref.ref(m[0]._parameters["weight"])
        pw.deactivate()
        assert module_param_ref() is not None  # still alive via model state
        del m, pw
        gc.collect()
        assert module_param_ref() is None  # GC freed it


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_rejects_empty_model(self) -> None:
        # Frozen but with no params or buffers — nothing to manage.
        class Empty(nn.Module):
            pass
        m = Empty()
        with pytest.raises(ValueError, match="at least one parameter"):
            _make_model_offloader(m)

    def test_accepts_buffer_only_module(self) -> None:
        # A module with only registered buffers (no frozen params) is a
        # legitimate target — common for things like RoPE position tables
        # or sinusoidal embeddings. ModelOffloader should pin the buffers
        # and behave as a no-op for params.
        class BufferOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("table", torch.randn(8, 4))

        m = BufferOnly()
        store = ModelOffloaderStore.from_module(m)
        pw = store.bind(m)
        try:
            assert store.cache_bytes == 8 * 4 * 4  # float32
            assert m.table.is_pinned()
        finally:
            pw.deactivate()

    def test_constructor_does_not_call_module_to(self) -> None:
        class Guarded(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4), requires_grad=False)

            def to(self, *args, **kwargs):
                raise AssertionError("constructor must pin directly")

        m = Guarded()
        pw = _make_model_offloader(m)
        try:
            assert m.weight.is_pinned()
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
        the rarer case that the original ModelOffloader silently broke
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
        pw = _make_model_offloader(m)
        try:
            # Exactly one unique pinned parameter for the tied weight.
            assert _unique_pinned_param_count(pw) == 1
            assert pw.param_names == {"embed.weight", "head.weight"}
            # After construction, both names reference the same Parameter
            # object, preserving tying at the strongest level.
            assert m.embed._parameters["weight"] is m.head._parameters["weight"]
            pinned = pinned_component(pw)._instance.params["embed.weight"]
            # The materialized CPU wrapper is installed onto the module (the
            # instance no longer caches it); both tied leaves share it.
            cpu_param = m.embed.weight
            assert m.head.weight is cpu_param
            assert cpu_param.data_ptr() == pinned.make_cpu_param().data_ptr()
        finally:
            pw.deactivate()

    def test_distinct_params_sharing_storage_dedupe(self) -> None:
        m, _, _ = self._make_distinct_param_tied_model()
        pw = _make_model_offloader(m)
        try:
            assert _unique_pinned_param_count(pw) == 1
            assert pw.param_names == {"a", "b"}
            # Both module entries now reference the same Parameter object.
            assert m._parameters["a"] is m._parameters["b"]
        finally:
            pw.deactivate()

    def test_zero_sized_same_parameter_under_two_names_dedupes(self) -> None:
        p = nn.Parameter(torch.empty(0), requires_grad=False)
        m = nn.Module()
        m.a = p
        m.b = p

        pw = _make_model_offloader(m)
        try:
            assert _unique_pinned_param_count(pw) == 1
            assert m._parameters["a"] is m._parameters["b"]
        finally:
            pw.deactivate()

    def test_cache_bytes_counts_tied_once(self) -> None:
        m, _, _ = self._make_tied_model()
        store = ModelOffloaderStore.from_module(m)
        pw = store.bind(m)
        try:
            # 32 * 16 * 4 (float32 default) = 2048 bytes for one pinned param.
            # If the dedup were broken this would double.
            assert store.cache_bytes == 32 * 16 * 4
        finally:
            pw.deactivate()

    @CUDA
    def test_tied_params_share_gpu_storage_on_activate(self) -> None:
        m, embed, head = self._make_tied_model()
        pw = _make_model_offloader(m)
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
        pw = _make_model_offloader(m)
        try:
            with pw.use("cuda"):
                # Registry identity comparison; the local `a` / `b` refs are
                # the now-orphaned originals (replaced in module registries).
                assert m._parameters["a"].is_cuda
                assert m._parameters["b"].is_cuda
                assert m._parameters["a"].data.data_ptr() == m._parameters["b"].data.data_ptr()
                assert m._parameters["a"] is m._parameters["b"]
        finally:
            pw.deactivate()

    @CUDA
    def test_cuda_origin_tied_params_dedupe_before_cpu_pin(self) -> None:
        shared = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
        m = nn.Module()
        m.a = nn.Parameter(shared, requires_grad=False)
        m.b = nn.Parameter(shared, requires_grad=False)

        pw = _make_model_offloader(m)
        try:
            assert _unique_pinned_param_count(pw) == 1
            assert m._parameters["a"] is m._parameters["b"]
            assert m.a.is_pinned()
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
        pw = _make_model_offloader(m)
        try:
            assert _unique_pinned_param_count(pw) == 1
            assert pw.param_names == {"a.weight", "b.weight"}
            assert m.a._parameters["weight"] is m.b._parameters["weight"]
        finally:
            pw.deactivate()

    def test_aliased_buffer(self) -> None:
        # Two distinct submodules sharing the same buffer tensor. The
        # default named_buffers() walks only one path; using
        # remove_duplicate=False plus tensor-id dedup ensures both
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
        pw = _make_model_offloader(m)
        try:
            # One pinned buffer backing covers both alias paths.
            assert _unique_pinned_buffer_count(pw) == 1
            assert pw.buffer_names == {"a.buf", "b.buf"}
            pinned_buffer = pinned_component(pw)._instance.buffers["a.buf"]
            assert pinned_buffer.tensor.is_pinned()
            # Both module entries reference the SAME pinned tensor.
            assert m.a.buf is pinned_buffer.tensor
            assert m.b.buf is pinned_buffer.tensor
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
        pw = _make_model_offloader(m)
        try:
            assert _unique_pinned_buffer_count(pw) == 1
            pinned_buffer = pinned_component(pw)._instance.buffers["a.buf"]
            assert m.a.buf is pinned_buffer.tensor
            assert m.b.buf is pinned_buffer.tensor
        finally:
            pw.deactivate()

    @CUDA
    def test_cuda_origin_aliased_buffer_dedupes_before_cpu_pin(self) -> None:
        shared_buf = torch.randn(4, device="cuda")

        class Inner(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.register_buffer("buf", b)
                self.weight = nn.Parameter(
                    torch.randn(2, device="cuda"),
                    requires_grad=False,
                )

        m = nn.Module()
        m.a = Inner(shared_buf)
        m.b = Inner(shared_buf)

        pw = _make_model_offloader(m)
        try:
            assert _unique_pinned_buffer_count(pw) == 1
            pinned_buffer = pinned_component(pw)._instance.buffers["a.buf"]
            assert m.a.buf is pinned_buffer.tensor
            assert m.b.buf is pinned_buffer.tensor
            assert m.a.buf.is_pinned()
        finally:
            pw.deactivate()


class TestMixedTrainableFrozenTied:
    def test_raises_when_tied_group_has_mixed_grad(self) -> None:
        # Two distinct Parameter objects sharing storage, one trainable
        # and one frozen. One pinned backing cannot preserve both
        # requires_grad values, so the store rejects the storage group.
        shared = torch.randn(8, dtype=torch.bfloat16)
        a = nn.Parameter(shared, requires_grad=True)
        b = nn.Parameter(shared, requires_grad=False)
        m = nn.Module()
        m.a = a
        m.b = b
        with pytest.raises(ValueError, match="mixed requires_grad"):
            _make_model_offloader(m)

class TestZeroSizedParams:
    def test_zero_sized_params_do_not_collapse(self) -> None:
        # Empty tensors all share data_ptr()==0; they must not dedupe
        # into one pinned param binding.
        m = nn.Module()
        m.a = nn.Parameter(torch.empty(0), requires_grad=False)
        m.b = nn.Parameter(torch.empty(0), requires_grad=False)
        # Need at least one non-empty frozen param so the constructor doesn't
        # reject the model. The empties should each be their own entry.
        m.c = nn.Parameter(torch.randn(4), requires_grad=False)
        pw = _make_model_offloader(m)
        try:
            # 3 entries: a, b, c — empties did not collapse.
            assert _unique_pinned_param_count(pw) == 3
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
        # The bug Codex found: the original ModelOffloader did
        # `p.data = binding.cpu_param.data` which is a no-op for quanto.
        # The fix: swap module._parameters[leaf] = binding.cpu_param.
        # Verify the model now references pinned _data storage.
        m = self._make_quanto_model()
        original_data_ptr = m.weight._data.data_ptr()
        pw = _make_model_offloader(m)
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
        pw = _make_model_offloader(m)
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
        pw = _make_model_offloader(m)
        pw.deactivate()
        # Quanto wrapper still on CPU after deactivate.
        assert m.weight._data.is_pinned()
        assert m.weight._scale.is_pinned()


# ---------------------------------------------------------------------------
# cache_bytes accounting
# ---------------------------------------------------------------------------


class TestCacheBytes:
    def test_cache_bytes_positive(self) -> None:
        store = ModelOffloaderStore.from_module(_make_simple_model())
        assert store.cache_bytes > 0

    def test_cache_bytes_includes_buffers(self) -> None:
        m = nn.Module()
        m.weight = nn.Parameter(torch.randn(4, 4), requires_grad=False)
        m.register_buffer("running", torch.randn(8))

        store = ModelOffloaderStore.from_module(m)

        expected = (4 * 4 * 4) + (8 * 4)
        assert store.cache_bytes == expected
