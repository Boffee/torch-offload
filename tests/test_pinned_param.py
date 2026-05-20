"""Tests for ``torch_offload.pinned_param.PinnedParam``."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

from torch_offload.pinned_bindings import (
    PinnedBuffer,
    PinnedBufferBinding,
    PinnedBufferTarget,
    PinnedModuleBinding,
    PinnedModuleTarget,
    PinnedParamBinding,
    PinnedParamTarget,
)
from torch_offload.pinned_param import PinnedParam
from torch_offload.slots import BufferSlot, ParamSlot
from torch_offload.tensor_adapters import (
    DequantRequantCopyIntoTensorAdapter,
    DequantRequantTensorAdapter,
    TensorCopyIntoAdapter,
)
from torch_offload.tensor_adapter_factory import select_adapter, storage_key

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _FakePinnedParam:
    def __init__(
        self,
        name: str,
        *,
        cache_bytes: int = 0,
        requires_grad: bool = False,
        target_data: torch.Tensor | None = None,
    ) -> None:
        self.name = name
        self.allocated = False
        self.copied = False
        self.copied_back = False
        self._cache_bytes = cache_bytes
        self._target_data = target_data
        self.requires_grad = requires_grad

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes

    def allocate_gpu_storage(self, device: torch.device) -> object:
        self.allocated = True
        return {"name": self.name, "device": device}

    def make_cpu_param(self) -> nn.Parameter:
        return nn.Parameter(torch.empty(0), requires_grad=self.requires_grad)

    def make_gpu_param(self, target_state: object) -> nn.Parameter:
        del target_state
        data = (
            torch.empty(0)
            if self._target_data is None
            else self._target_data.clone()
        )
        return nn.Parameter(data, requires_grad=self.requires_grad)

    def copy_to_gpu(
        self,
        target_state: object,
        *,
        non_blocking: bool = False,
    ) -> None:
        del target_state, non_blocking
        self.copied = True

    def copy_to_cpu(
        self,
        target_state: object,
        *,
        non_blocking: bool = False,
    ) -> None:
        del target_state, non_blocking
        self.copied_back = True


class _FakePinnedBuffer:
    def __init__(
        self,
        name: str,
        *,
        cache_bytes: int = 0,
        tensor: torch.Tensor | None = None,
    ) -> None:
        self.name = name
        self.tensor = tensor if tensor is not None else torch.empty(0)
        self._cache_bytes = cache_bytes

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes


def _target_binding(
    params: list[tuple[str, PinnedParam]],
    buffers: list[tuple[str, PinnedBuffer]] | None = None,
) -> PinnedModuleBinding:
    return PinnedModuleBinding(
        param_bindings=[
            PinnedParamBinding(
                name=name,
                pinned=pinned,
                slots=[],
                cpu_param=pinned.make_cpu_param(),
            )
            for name, pinned in params
        ],
        buffer_bindings=[
            PinnedBufferBinding(name=name, pinned=pinned, slots=[])
            for name, pinned in (buffers or [])
        ],
    )


def test_module_target_snapshots_binding_layout_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_empty_like = torch.empty_like

    def fake_empty_like(
        tensor: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        assert tensor.numel() == 0
        assert device == torch.device("cuda")
        return original_empty_like(tensor)

    monkeypatch.setattr(torch, "empty_like", fake_empty_like)
    param = _FakePinnedParam("w")
    buffer = _FakePinnedBuffer("buf")
    binding = cast(
        PinnedModuleBinding,
        _target_binding(
            [("w", cast(PinnedParam, param))],
            [("buf", cast(PinnedBuffer, buffer))],
        ),
    )

    target = binding.allocate_target(torch.device("cuda"))
    binding.param_bindings.clear()
    binding.buffer_bindings.clear()

    assert param.allocated
    assert target.param_target("w").param is not None
    assert target.buffer_target("buf").tensor is not None


def test_module_target_rejects_non_cuda_before_child_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_like_calls = 0

    def fake_empty_like(
        tensor: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        del tensor, device
        nonlocal empty_like_calls
        empty_like_calls += 1
        raise AssertionError("buffer allocation should not run")

    monkeypatch.setattr(torch, "empty_like", fake_empty_like)
    param = _FakePinnedParam("w")
    buffer = _FakePinnedBuffer("buf")
    binding = _target_binding(
        [("w", cast(PinnedParam, param))],
        [("buf", cast(PinnedBuffer, buffer))],
    )

    with pytest.raises(ValueError, match="requires a CUDA device"):
        binding.allocate_target(torch.device("cpu"))

    assert not param.allocated
    assert empty_like_calls == 0


def test_module_target_uses_binding_name_as_storage_key() -> None:
    param = _FakePinnedParam("pinned-name")
    binding = PinnedParamBinding(
        name="target-name",
        pinned=cast(PinnedParam, param),
        slots=[],
        cpu_param=param.make_cpu_param(),
    )
    module_binding = PinnedModuleBinding(
        param_bindings=[binding],
        buffer_bindings=[],
    )

    target = module_binding.allocate_target(torch.device("cuda"))

    module_binding.load_to_target(target)
    assert target.param_target("target-name").param is not None
    assert param.allocated
    assert param.copied


# ---------------------------------------------------------------------------
# PinnedParam basic correctness
# ---------------------------------------------------------------------------


class TestPinnedParam:
    def test_select_adapter_returns_adapter_for_plain_tensor(self) -> None:
        first = select_adapter(torch.randn(1))
        second = select_adapter(torch.randn(2))

        assert type(first) is type(second)

    def test_regular_storage_key_includes_device(self) -> None:
        t = torch.randn(2, 3)

        assert storage_key(t)[:2] == ("regular", t.device)

    def test_non_quanto_pin_and_load(self) -> None:
        p = nn.Parameter(torch.randn(8, 16, dtype=torch.bfloat16), requires_grad=False)
        pinned_param = PinnedParam(p)
        cpu_param = pinned_param.make_cpu_param()
        other_cpu_param = pinned_param.make_cpu_param()

        # make_cpu_param wraps a plain pinned tensor — no quanto subclass.
        assert type(cpu_param.data) is torch.Tensor
        assert cpu_param.data.is_pinned()
        assert cpu_param.data.shape == p.shape
        # CPU params are distinct wrappers over the same pinned host buffer,
        # not a second clone — callers slot-replace at this and rely on
        # the storage staying alive for the pinned parameter's lifetime.
        assert cpu_param is not other_cpu_param
        assert cpu_param.data.data_ptr() == pinned_param.pinned_state.data.data_ptr()
        assert other_cpu_param.data.data_ptr() == pinned_param.pinned_state.data.data_ptr()
        # Low-peak construction repoints the source Parameter to the
        # pinned backing without making PinnedParam own that wrapper.
        assert p.data.data_ptr() == pinned_param.pinned_state.data.data_ptr()

    @CUDA
    def test_load_to_gpu_non_quanto(self) -> None:
        p = nn.Parameter(torch.randn(4, 8, dtype=torch.bfloat16), requires_grad=False)
        pinned_param = PinnedParam(p)
        gpu = pinned_param.load_to_gpu(torch.device("cuda"))
        assert gpu.is_cuda
        assert gpu.shape == p.shape
        torch.cuda.synchronize()
        assert torch.equal(gpu.cpu(), pinned_param.make_cpu_param().data)

    @CUDA
    def test_pool_pattern_allocate_and_copy(self) -> None:
        # Mirrors how PinnedModuleTarget uses PinnedParam: allocate GPU
        # storage once, then copy_to_gpu in place on each load.
        p = nn.Parameter(torch.randn(16, dtype=torch.bfloat16), requires_grad=False)
        pinned_param = PinnedParam(p)
        device = torch.device("cuda")
        gpu_state = pinned_param.allocate_gpu_storage(device)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        assert gpu_param.is_cuda
        # First copy
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        torch.cuda.synchronize()
        cpu_param = pinned_param.make_cpu_param()
        assert torch.equal(gpu_state.data.cpu(), cpu_param.data)
        # Mutate pinned source and re-copy — gpu state should track.
        new_vals = torch.randn(16, dtype=torch.bfloat16, pin_memory=True)
        cpu_param.data.copy_(new_vals)
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(gpu_state.data.cpu(), new_vals)
        # Stable storage — gpu_param wraps the same GPU bytes as gpu_state.
        # PinnedModuleTarget relies on this: build the Parameter wrapper once at target
        # construction, mutate underlying storage in place on each load.
        assert gpu_param.data_ptr() == gpu_state.data.data_ptr()

    def test_contiguous_format_forced(self) -> None:
        # A view of a transposed tensor is non-contiguous. clone() with
        # contiguous_format normalizes it; pinned data must be 1-D
        # contiguous so downstream callers can rely on it.
        base = torch.randn(8, 16, dtype=torch.bfloat16)
        non_contig = base.t()
        assert not non_contig.is_contiguous()
        p = nn.Parameter(non_contig, requires_grad=False)
        pinned_param = PinnedParam(p)
        cpu_param = pinned_param.make_cpu_param()
        assert cpu_param.data.is_contiguous()
        assert cpu_param.data.is_pinned()

    @CUDA
    def test_slot_param_identity_stable_across_loads(self) -> None:
        # PinnedModuleTarget caches the Parameter wrapping its GPU storage;
        # load_params must not churn those wrappers. Module slots observe a
        # stable object across reloads — the whole point of the pooled
        # target pattern over per-load allocation.
        p1 = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        p2 = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        block = [("a", PinnedParam(p1)), ("b", PinnedParam(p2))]
        binding = _target_binding(block)
        target = binding.allocate_target(torch.device("cuda"))

        binding.load_to_target(target, non_blocking=False)
        a_first = target.param_target("a").param
        b_first = target.param_target("b").param
        torch.cuda.synchronize()
        binding.load_to_target(target, non_blocking=False)
        assert target.param_target("a").param is a_first
        assert target.param_target("b").param is b_first
        torch.cuda.synchronize()
        binding.load_to_target(target, non_blocking=False)
        assert target.param_target("a").param is a_first
        assert target.param_target("b").param is b_first

    @CUDA
    def test_module_target_rejects_duplicate_param_names(self) -> None:
        p1 = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        p2 = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        block = [("w", PinnedParam(p1)), ("w", PinnedParam(p2))]

        with pytest.raises(
            ValueError,
            match="duplicate 'w'",
        ):
            _target_binding(block).allocate_target(torch.device("cuda"))

    @CUDA
    def test_module_target_rejects_duplicate_buffer_names(self) -> None:
        buffers = [
            ("buf", PinnedBuffer.clone(torch.randn(8))),
            ("buf", PinnedBuffer.clone(torch.randn(8))),
        ]

        with pytest.raises(
            ValueError,
            match="duplicate 'buf'",
        ):
            _target_binding([], buffers).allocate_target(torch.device("cuda"))

    @CUDA
    def test_slot_buffer_identity_stable_across_loads(self) -> None:
        pinned = PinnedBuffer.clone(torch.randn(8))
        binding = _target_binding([], [("buf", pinned)])
        target = binding.allocate_target(torch.device("cuda"))

        binding.load_to_target(target, non_blocking=False)
        first = target.buffer_target("buf").tensor
        torch.cuda.synchronize()
        pinned.tensor.copy_(torch.randn(8))
        binding.load_to_target(target, non_blocking=False)
        second = target.buffer_target("buf").tensor
        torch.cuda.synchronize()

        assert second is first
        torch.testing.assert_close(first.cpu(), pinned.tensor)

    @CUDA
    def test_slot_hook_mutates_stable_param_in_place(self) -> None:
        p = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        pinned_param = PinnedParam(p)
        block = [("w", pinned_param)]
        binding = _target_binding(block)
        target = binding.allocate_target(torch.device("cuda"))
        binding.load_to_target(target, non_blocking=False)
        base_param = target.param_target("w").param

        def hook(param: nn.Parameter) -> None:
            param.data.add_(1)

        binding.load_to_target(
            target,
            post_copy_hooks={id(binding.param_bindings[0]): hook},
            non_blocking=False,
        )
        torch.cuda.synchronize()
        cpu_param = pinned_param.make_cpu_param()
        torch.testing.assert_close(
            base_param.detach().cpu(),
            cpu_param.detach() + 1,
        )

        binding.load_to_target(target, non_blocking=False)
        assert target.param_target("w").param is base_param
        torch.cuda.synchronize()
        torch.testing.assert_close(
            base_param.detach().cpu(),
            cpu_param.detach(),
        )


class TestPinnedBindings:
    def test_module_binding_cache_bytes_dedupes_repeated_backing(
        self,
    ) -> None:
        pinned_param = cast(
            PinnedParam,
            _FakePinnedParam("shared-param", cache_bytes=64),
        )
        pinned_buffer = cast(
            PinnedBuffer,
            _FakePinnedBuffer("shared-buffer", cache_bytes=16),
        )
        module_binding = _target_binding(
            [("weight", pinned_param), ("tied_weight", pinned_param)],
            [("table", pinned_buffer), ("tied_table", pinned_buffer)],
        )

        assert module_binding.cache_bytes == 80

    def test_module_binding_cache_bytes_are_not_cross_binding_ownership(
        self,
    ) -> None:
        pinned_param = cast(
            PinnedParam,
            _FakePinnedParam("shared-param", cache_bytes=64),
        )
        pinned_buffer = cast(
            PinnedBuffer,
            _FakePinnedBuffer("shared-buffer", cache_bytes=16),
        )
        first = _target_binding(
            [("weight", pinned_param)],
            [("table", pinned_buffer)],
        )
        second = _target_binding(
            [("weight", pinned_param)],
            [("table", pinned_buffer)],
        )

        unique_backing_bytes = pinned_param.cache_bytes + pinned_buffer.cache_bytes

        assert first.cache_bytes == unique_backing_bytes
        assert second.cache_bytes == unique_backing_bytes
        assert first.cache_bytes + second.cache_bytes == (
            unique_backing_bytes * 2
        )

    def test_param_binding_target_wraps_adapter_state(self) -> None:
        pinned = _FakePinnedParam("weight")
        binding = PinnedParamBinding(
            name="weight",
            pinned=cast(PinnedParam, pinned),
            slots=[],
            cpu_param=pinned.make_cpu_param(),
        )

        target = binding.allocate_target(torch.device("cuda"))

        assert isinstance(target, PinnedParamTarget)
        assert target.param.requires_grad is False
        assert pinned.allocated

    def test_param_binding_copies_to_target(self) -> None:
        pinned = _FakePinnedParam("weight")
        binding = PinnedParamBinding(
            name="weight",
            pinned=cast(PinnedParam, pinned),
            slots=[],
            cpu_param=pinned.make_cpu_param(),
        )
        target = binding.allocate_target(torch.device("cuda"))

        binding.copy_to_target(target, non_blocking=True)

        assert pinned.copied

    def test_param_binding_rejects_copy_from_target_for_frozen_binding(
        self,
    ) -> None:
        pinned = _FakePinnedParam("weight", requires_grad=False)
        binding = PinnedParamBinding(
            name="weight",
            pinned=cast(PinnedParam, pinned),
            slots=[],
            cpu_param=pinned.make_cpu_param(),
        )
        target = binding.allocate_target(torch.device("cuda"))

        with pytest.raises(RuntimeError, match="only valid for trainable"):
            binding.copy_from_target(target, non_blocking=True)

        assert not pinned.copied_back

    def test_param_binding_copies_from_target_for_trainable_binding(
        self,
    ) -> None:
        pinned = _FakePinnedParam("weight", requires_grad=True)
        binding = PinnedParamBinding(
            name="weight",
            pinned=cast(PinnedParam, pinned),
            slots=[],
            cpu_param=pinned.make_cpu_param(),
        )
        target = binding.allocate_target(torch.device("cuda"))

        binding.copy_from_target(target, non_blocking=True)

        assert pinned.copied_back

    def test_buffer_binding_copies_to_target(self) -> None:
        pinned = _FakePinnedBuffer(
            "table",
            tensor=torch.tensor([1.0, 2.0]),
        )
        binding = PinnedBufferBinding(
            name="table",
            pinned=cast(PinnedBuffer, pinned),
            slots=[],
        )

        target = binding.allocate_target(torch.device("cpu"))
        assert isinstance(target, PinnedBufferTarget)
        assert target.tensor.shape == pinned.tensor.shape

        target.tensor.zero_()
        binding.copy_to_target(target)

        torch.testing.assert_close(target.tensor, pinned.tensor)

    def test_module_binding_load_to_target_copies_before_setting_slots(
        self,
    ) -> None:
        parent = nn.Module()
        parent.weight = nn.Parameter(
            torch.tensor([1.0, 2.0]), requires_grad=False,
        )
        slot = ParamSlot("weight", parent, "weight")
        pinned = _FakePinnedParam(
            "weight",
            target_data=torch.tensor([3.0, 4.0]),
        )
        binding = PinnedParamBinding(
            name="weight",
            pinned=cast(PinnedParam, pinned),
            slots=[slot],
            cpu_param=pinned.make_cpu_param(),
        )
        module_binding = PinnedModuleBinding(
            param_bindings=[binding],
            buffer_bindings=[],
        )
        binding.restore_pinned()
        slot_param = parent.weight

        target = module_binding.allocate_target(torch.device("cuda"))
        target_param = target.param_target("weight").param
        hook_calls: list[nn.Parameter] = []

        def hook(param: nn.Parameter) -> None:
            assert parent.weight is slot_param
            hook_calls.append(param)
            param.data.add_(1)

        module_binding.load_to_target(
            target,
            post_copy_hooks={id(binding): hook},
            non_blocking=True,
        )

        assert hook_calls == [target_param]
        assert parent.weight is target_param
        torch.testing.assert_close(
            target_param.data,
            torch.tensor([4.0, 5.0]),
        )

    def test_module_binding_load_to_target_preserves_trainable_wrapper(
        self,
    ) -> None:
        parent = nn.Module()
        parent.weight = nn.Parameter(
            torch.tensor([1.0, 2.0]), requires_grad=True,
        )
        slot = ParamSlot("weight", parent, "weight")
        pinned = _FakePinnedParam(
            "weight",
            requires_grad=True,
            target_data=torch.tensor([7.0, 8.0]),
        )
        binding = PinnedParamBinding(
            name="weight",
            pinned=cast(PinnedParam, pinned),
            slots=[slot],
            cpu_param=pinned.make_cpu_param(),
        )
        module_binding = PinnedModuleBinding(
            param_bindings=[binding],
            buffer_bindings=[],
        )
        binding.restore_pinned()
        slot_param = parent.weight

        target = module_binding.allocate_target(torch.device("cuda"))
        target_param = target.param_target("weight").param
        module_binding.load_to_target(target)

        assert parent.weight is slot_param
        assert parent.weight.data.data_ptr() == target_param.data.data_ptr()
        torch.testing.assert_close(parent.weight.data, target_param.data)

    def test_module_binding_loads_buffer_to_target_and_restore_pinned(self) -> None:
        parent = nn.Module()
        original = torch.tensor([1.0, 2.0])
        parent.register_buffer("running", original, persistent=False)
        slot = BufferSlot("running", parent, "running", persistent=False)
        pinned = PinnedBuffer.clone(torch.tensor([5.0, 6.0]))
        binding = PinnedBufferBinding(
            name="running",
            pinned=pinned,
            slots=[slot],
        )
        module_binding = PinnedModuleBinding(
            param_bindings=[],
            buffer_bindings=[binding],
        )
        copied = torch.tensor([5.0, 6.0])
        target = PinnedModuleTarget(
            device=torch.device("cuda"),
            param_targets={},
            buffer_targets={"running": PinnedBufferTarget(copied)},
        )

        assert binding.name == "running"
        assert pinned.tensor.is_pinned()
        assert pinned.cache_bytes == pinned.tensor.numel() * pinned.tensor.element_size()
        assert parent.running is original

        module_binding.load_to_target(target)

        assert parent.running is copied
        assert "running" in parent._non_persistent_buffers_set

        active = torch.tensor([9.0, 10.0])
        binding.set_slots(active)
        assert parent.running is active

        binding.restore_pinned()

        assert parent.running is pinned.tensor
        assert "running" in parent._non_persistent_buffers_set


# ---------------------------------------------------------------------------
# Quanto path — only nontrivial branch in PinnedParam
# ---------------------------------------------------------------------------


class TestPinnedParamQuanto:
    def test_quanto_storage_key_includes_inner_devices(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-32, 32, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )

        key = storage_key(qt)
        assert key[1] == qt._data.device
        assert key[7] == qt._scale.device

    def test_quanto_adapter_dequant_requant_capability(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-32, 32, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )

        adapter = select_adapter(qt)
        assert isinstance(adapter, DequantRequantCopyIntoTensorAdapter)
        assert isinstance(adapter, DequantRequantTensorAdapter)
        assert isinstance(adapter, TensorCopyIntoAdapter)
        dense = adapter.dequantize(qt)
        assert type(dense) is torch.Tensor
        assert dense.dtype == torch.float32
        assert dense.shape == qt.shape

        requantized = adapter.requantize(dense, like=qt)
        assert isinstance(requantized, WeightQBytesTensor)
        assert requantized.qtype is quanto.qint8
        assert requantized.axis == 0
        assert tuple(requantized.size()) == (rows, cols)
        torch.testing.assert_close(requantized._data, data)
        torch.testing.assert_close(requantized._scale, scale)

        updated = dense + 1
        expected_packed = (
            updated / scale.to(torch.float32)
        ).round().clamp(-128, 127).to(torch.int8)
        updated_qt = adapter.requantize(updated, like=qt)
        original_scale_ptr = qt._scale.data_ptr()
        adapter.copy_into(updated_qt, target=qt)
        torch.testing.assert_close(qt._data, expected_packed)
        torch.testing.assert_close(qt._scale, scale)
        assert qt._scale.data_ptr() == original_scale_ptr

    def test_pin_decomposes_data_and_scale(self) -> None:
        # Quanto WeightQBytesTensor must be decomposed into _data + _scale
        # and the CPU wrapper reconstructed from the pinned tensors.
        # A naive tensor.clone() would silently dequantize via the dispatch
        # fallback — that bug is the reason pinned_param.py exists.
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )
        p = nn.Parameter(qt, requires_grad=False)
        pinned_param = PinnedParam(p)

        # make_cpu_param wraps a quanto tensor whose _data and _scale are
        # pinned, contiguous, and carry the original quant metadata.
        cpu_param = pinned_param.make_cpu_param()
        assert isinstance(cpu_param.data, WeightQBytesTensor)
        qt_pinned = cpu_param.data
        assert qt_pinned._data.is_pinned()
        assert qt_pinned._data.is_contiguous()
        assert qt_pinned._data.dtype == torch.int8
        assert qt_pinned._scale.is_pinned()
        assert qt_pinned.qtype is quanto.qint8
        assert qt_pinned.axis == 0
        assert tuple(qt_pinned.size()) == (rows, cols)
        assert qt_pinned.stride() == (cols, 1)
        assert getattr(qt_pinned, "activation_qtype", None) is None
        # Zero-copy: the wrapper's _data and _scale point at the same
        # pinned host buffers the adapter pinned, not separate clones.
        assert qt_pinned._data.data_ptr() == pinned_param.pinned_state.data.data_ptr()
        assert qt_pinned._scale.data_ptr() == pinned_param.pinned_state.scale.data_ptr()

    @CUDA
    def test_load_to_gpu_round_trip(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )
        p = nn.Parameter(qt, requires_grad=False)
        pinned_param = PinnedParam(p)

        gpu_param = pinned_param.load_to_gpu(torch.device("cuda"))
        torch.cuda.synchronize()
        assert isinstance(gpu_param.data, WeightQBytesTensor)
        assert gpu_param.data._data.is_cuda
        assert gpu_param.data._scale.is_cuda
        qt_pinned = pinned_param.make_cpu_param().data
        assert torch.equal(gpu_param.data._data.cpu(), qt_pinned._data)
        assert torch.equal(gpu_param.data._scale.cpu(), qt_pinned._scale)


# ---------------------------------------------------------------------------
# requires_grad propagation through the wrapper builders
# ---------------------------------------------------------------------------


class TestRequiresGradPropagation:
    """The pinned parameter captures the source param's ``requires_grad`` at
    construction time and threads it through to ``make_cpu_param`` /
    ``gpu_param``. Frozen sources get historic ``requires_grad=False``
    wrappers; trainable sources get ``True`` wrappers so consumers
    that DO use the wrapper objects (rather than ``.data``-swapping)
    see the right autograd flag."""

    def test_frozen_source_yields_frozen_cpu_param(self) -> None:
        p = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        pinned_param = PinnedParam(p)
        assert pinned_param.requires_grad is False
        assert pinned_param.make_cpu_param().requires_grad is False

    def test_trainable_source_yields_trainable_cpu_param(self) -> None:
        p = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=True)
        pinned_param = PinnedParam(p)
        assert pinned_param.requires_grad is True
        assert pinned_param.make_cpu_param().requires_grad is True

    @CUDA
    def test_trainable_source_yields_trainable_gpu_param(self) -> None:
        p = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=True)
        pinned_param = PinnedParam(p)
        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        assert gpu_param.requires_grad is True

    @CUDA
    def test_frozen_source_yields_frozen_gpu_param(self) -> None:
        p = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        pinned_param = PinnedParam(p)
        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        assert gpu_param.requires_grad is False


# ---------------------------------------------------------------------------
# copy_to_cpu — D2H counterpart to copy_to_gpu
# ---------------------------------------------------------------------------


class TestCopyToCpu:
    """Symmetric D2H of GPU contents back into the pinned host state.
    Used at the optimizer-step boundary in trainable streaming: GPU
    weights got updated in place by ``optimizer.step()``, scatter the
    update back to the pinned clone so the next H2D reads it."""

    @CUDA
    def test_regular_round_trip(self) -> None:
        p = nn.Parameter(torch.randn(16, dtype=torch.bfloat16), requires_grad=False)
        original = p.data.clone()
        pinned_param = PinnedParam(p)
        device = torch.device("cuda")
        gpu_state = pinned_param.allocate_gpu_storage(device)
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        torch.cuda.synchronize()

        # Mutate GPU side as if optimizer.step had run there.
        new_vals_gpu = torch.randn(16, dtype=torch.bfloat16, device=device)
        gpu_state.data.copy_(new_vals_gpu)
        torch.cuda.synchronize()

        # D2H should overwrite the pinned host state with the new GPU contents.
        pinned_param.copy_to_cpu(gpu_state, non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(pinned_param.pinned_state.data, new_vals_gpu.cpu())
        # The pinned state has actually changed from the original.
        assert not torch.equal(pinned_param.pinned_state.data, original)

    @CUDA
    def test_regular_pinned_storage_identity_preserved(self) -> None:
        # The pinned-host buffer stays at the same address after D2H —
        # we're overwriting in place, not allocating a new tensor.
        # Callers that hold binding cpu_param.data references rely on this.
        p = nn.Parameter(torch.randn(16, dtype=torch.bfloat16), requires_grad=True)
        pinned_param = PinnedParam(p)
        original_ptr = pinned_param.pinned_state.data.data_ptr()
        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state)
        pinned_param.copy_to_cpu(gpu_state)
        torch.cuda.synchronize()
        assert pinned_param.pinned_state.data.data_ptr() == original_ptr

    @CUDA
    def test_quanto_round_trip(self) -> None:
        # Quanto D2H must write back BOTH _data (int8) and _scale, and
        # the quant metadata on the pinned wrapper must be unchanged.
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )
        p = nn.Parameter(qt, requires_grad=False)
        pinned_param = PinnedParam(p)
        device = torch.device("cuda")
        gpu_state = pinned_param.allocate_gpu_storage(device)
        pinned_param.copy_to_gpu(gpu_state)
        torch.cuda.synchronize()

        # Mutate GPU-side _data and _scale in place.
        new_data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8, device=device)
        new_scale = torch.rand(rows, 1, dtype=torch.bfloat16, device=device)
        gpu_state.data.copy_(new_data)
        gpu_state.scale.copy_(new_scale)
        torch.cuda.synchronize()

        pinned_param.copy_to_cpu(gpu_state)
        torch.cuda.synchronize()
        assert torch.equal(pinned_param.pinned_state.data, new_data.cpu())
        assert torch.equal(pinned_param.pinned_state.scale, new_scale.cpu())
        # Metadata untouched — qtype, axis, etc. live on the pinned
        # state and are not part of the GPU representation.
        assert pinned_param.pinned_state.qtype is quanto.qint8
        assert pinned_param.pinned_state.axis == 0

    @CUDA
    def test_gguf_lacks_cpu_round_trip_capability(self) -> None:
        # GGUF stores packed quantized bytes on CPU but dequantized
        # bf16 on GPU — D2H would require re-quantization, which isn't
        # implemented. Surface it as NotImplementedError, not a silent
        # corruption of the pinned packed bytes.
        gguf = pytest.importorskip("gguf")
        from torch_offload.gguf_adapter import GGUFWeight

        # Build minimal GGUF state directly via the adapter — avoids
        # needing a real .gguf file to load. Q4_0 has the simplest
        # block layout and is broadly supported.
        qt_value = int(gguf.GGMLQuantizationType.Q4_0)
        # Q4_0 packs 32 fp16 weights into an 18-byte block (2 scale + 16 quants).
        packed = torch.zeros(18, dtype=torch.uint8)
        gguf_t = GGUFWeight(packed, quant_type=qt_value)
        p = nn.Parameter(gguf_t, requires_grad=False)
        pinned_param = PinnedParam(p)
        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        with pytest.raises(NotImplementedError, match="CPU round-trip"):
            pinned_param.copy_to_cpu(gpu_state)
