"""Tests for ``torch_offload.pinned_param.PinnedParam``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torch_offload.pinned_buffer import PinnedBuffer
from torch_offload.pinned_param import PinnedParam
from torch_offload.tensor_adapters import (
    DequantRequantCopyIntoTensorAdapter,
    DequantRequantTensorAdapter,
    TensorCopyIntoAdapter,
)
from torch_offload.tensor_adapter_registry import select_adapter, tensor_id

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# PinnedParam basic correctness
# ---------------------------------------------------------------------------


class TestPinnedParam:
    def test_select_adapter_returns_adapter_for_plain_tensor(self) -> None:
        first = select_adapter(torch.randn(1))
        second = select_adapter(torch.randn(2))

        assert type(first) is type(second)

    def test_regular_tensor_id_includes_device(self) -> None:
        t = torch.randn(2, 3)

        assert tensor_id(t)[:2] == ("regular", t.device)

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
        # not a second clone — callers replace registry entries with this and rely on
        # the storage staying alive for the pinned parameter's lifetime.
        assert cpu_param is not other_cpu_param
        assert cpu_param.data.data_ptr() == pinned_param.pinned_state.data.data_ptr()
        assert other_cpu_param.data.data_ptr() == pinned_param.pinned_state.data.data_ptr()
        # Low-peak construction repoints the source Parameter to the
        # pinned backing without making PinnedParam own that wrapper.
        assert p.data.data_ptr() == pinned_param.pinned_state.data.data_ptr()

    @CUDA
    def test_allocate_copy_make_gpu_param_non_quanto(self) -> None:
        p = nn.Parameter(torch.randn(4, 8, dtype=torch.bfloat16), requires_grad=False)
        pinned_param = PinnedParam(p)
        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu = pinned_param.make_gpu_param(gpu_state)
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

class TestPinnedBuffer:
    def test_target_layout_matches_pinned_tensor_layout(self) -> None:
        source = torch.randn(2, 3).t()

        pinned = PinnedBuffer.clone(source)

        assert not source.is_contiguous()
        assert pinned.tensor.is_contiguous()
        assert pinned.target_layout == PinnedBuffer.target_layout_for(
            pinned.tensor,
        )


# ---------------------------------------------------------------------------
# Quanto path — only nontrivial branch in PinnedParam
# ---------------------------------------------------------------------------


class TestPinnedParamQuanto:
    def test_quanto_tensor_id_includes_inner_devices(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-32, 32, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )

        key = tensor_id(qt)
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
    def test_allocate_copy_make_gpu_param_quanto_round_trip(self) -> None:
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

        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
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
    def test_regular_pinned_tensor_identity_preserved(self) -> None:
        # The pinned-host buffer stays at the same address after D2H —
        # we're overwriting in place, not allocating a new tensor.
        # Callers that hold CPU Parameter wrapper references rely on this.
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
