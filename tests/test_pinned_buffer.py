"""Tests for ``torch_offload.pinned_buffer.PinnedParamBuffer``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torch_offload.pinned_buffer import PinnedParamBuffer

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# PinnedParamBuffer basic correctness
# ---------------------------------------------------------------------------


class TestPinnedParamBuffer:
    def test_non_quanto_pin_and_load(self) -> None:
        p = nn.Parameter(torch.randn(8, 16, dtype=torch.bfloat16), requires_grad=False)
        buf = PinnedParamBuffer("w", p)
        # cpu_param wraps a plain pinned tensor — no quanto subclass.
        assert type(buf.cpu_param.data) is torch.Tensor
        assert buf.cpu_param.data.is_pinned()
        assert buf.cpu_param.data.shape == p.shape
        # cpu_param is a zero-copy wrapper around the pinned host buffer,
        # not a second clone — callers slot-replace at this and rely on
        # the storage staying alive for the buffer's lifetime.
        assert buf.cpu_param.data.data_ptr() == buf.pinned_state.data.data_ptr()

    @CUDA
    def test_load_to_gpu_non_quanto(self) -> None:
        p = nn.Parameter(torch.randn(4, 8, dtype=torch.bfloat16), requires_grad=False)
        buf = PinnedParamBuffer("w", p)
        gpu = buf.load_to_gpu(torch.device("cuda"))
        assert gpu.is_cuda
        assert gpu.shape == p.shape
        torch.cuda.synchronize()
        assert torch.equal(gpu.cpu(), buf.cpu_param.data)

    @CUDA
    def test_pool_pattern_allocate_and_copy(self) -> None:
        # Mirrors how _GpuSlot uses PinnedParamBuffer: allocate GPU
        # storage once, then copy_to_gpu in place on each load.
        p = nn.Parameter(torch.randn(16, dtype=torch.bfloat16), requires_grad=False)
        buf = PinnedParamBuffer("w", p)
        device = torch.device("cuda")
        gpu_state = buf.allocate_gpu_storage(device)
        gpu_param = buf.make_gpu_param(gpu_state)
        assert gpu_param.is_cuda
        # First copy
        buf.copy_to_gpu(gpu_state, non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(gpu_state.data.cpu(), buf.cpu_param.data)
        # Mutate pinned source and re-copy — gpu state should track.
        new_vals = torch.randn(16, dtype=torch.bfloat16, pin_memory=True)
        buf.cpu_param.data.copy_(new_vals)
        buf.copy_to_gpu(gpu_state, non_blocking=True)
        torch.cuda.synchronize()
        assert torch.equal(gpu_state.data.cpu(), new_vals)
        # Stable storage — gpu_param wraps the same GPU bytes as gpu_state.
        # _GpuSlot relies on this: build the Parameter wrapper once at slot
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
        buf = PinnedParamBuffer("w", p)
        assert buf.cpu_param.data.is_contiguous()
        assert buf.cpu_param.data.is_pinned()

    @CUDA
    def test_slot_param_identity_stable_across_loads(self) -> None:
        # _GpuSlot caches the Parameter wrapping its GPU storage; copy_from
        # must not churn that wrapper. Hooks repointing submod._parameters
        # at slot.get_param() observe a stable object across reloads — the
        # whole point of the pool-slot pattern over per-load allocation.
        from torch_offload.streamed_weights import _GpuSlot

        p1 = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        p2 = nn.Parameter(torch.randn(8, dtype=torch.bfloat16), requires_grad=False)
        block = [PinnedParamBuffer("a", p1), PinnedParamBuffer("b", p2)]
        slot = _GpuSlot(block, torch.device("cuda"))

        a_first = slot.get_param("a")
        b_first = slot.get_param("b")
        slot.copy_from(block, non_blocking=False)
        torch.cuda.synchronize()
        assert slot.get_param("a") is a_first
        assert slot.get_param("b") is b_first
        slot.copy_from(block, non_blocking=False)
        torch.cuda.synchronize()
        assert slot.get_param("a") is a_first
        assert slot.get_param("b") is b_first


# ---------------------------------------------------------------------------
# Quanto path — only nontrivial branch in PinnedParamBuffer
# ---------------------------------------------------------------------------


class TestPinnedParamBufferQuanto:
    def test_pin_decomposes_data_and_scale(self) -> None:
        # Quanto WeightQBytesTensor must be decomposed into _data + _scale
        # and the cpu_param wrapper reconstructed from the pinned tensors.
        # A naive tensor.clone() would silently dequantize via the dispatch
        # fallback — that bug is the reason pinned_buffer.py exists.
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )
        p = nn.Parameter(qt, requires_grad=False)
        buf = PinnedParamBuffer("w", p)

        # cpu_param wraps a quanto tensor whose _data and _scale are
        # pinned, contiguous, and carry the original quant metadata.
        assert isinstance(buf.cpu_param.data, WeightQBytesTensor)
        qt_pinned = buf.cpu_param.data
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
        assert qt_pinned._data.data_ptr() == buf.pinned_state.data.data_ptr()
        assert qt_pinned._scale.data_ptr() == buf.pinned_state.scale.data_ptr()

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
        buf = PinnedParamBuffer("w", p)

        gpu_param = buf.load_to_gpu(torch.device("cuda"))
        torch.cuda.synchronize()
        assert isinstance(gpu_param.data, WeightQBytesTensor)
        assert gpu_param.data._data.is_cuda
        assert gpu_param.data._scale.is_cuda
        qt_pinned = buf.cpu_param.data
        assert torch.equal(gpu_param.data._data.cpu(), qt_pinned._data)
        assert torch.equal(gpu_param.data._scale.cpu(), qt_pinned._scale)
