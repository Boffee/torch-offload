"""Tests for TorchAO Int8 (``Int8Tensor``) adapter integration."""

from __future__ import annotations


import pytest
import torch
from torch import nn

from torch_offload import (
    LoRA,
    LoRATransform,
    ModelOffloader,
    ScaledLoRAFactor,
    StreamConfig,
    merge_lora,
)
from torch_offload.int8_adapter import Int8Adapter
from torch_offload.pinned_param import PinnedParam
from torch_offload.streamed_component import _param_target_layout
from torch_offload.tensor_adapter_registry import select_adapter, tensor_id
from tests.conftest import activated_model

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_model_offloader(
    model: nn.Module,
    *,
    blocks_attr: list[str] = [],
    stream_trainable_weights: bool = False,
) -> ModelOffloader:
    return ModelOffloader.from_module(
        model,
        blocks_attr=blocks_attr,
        stream_trainable_weights=stream_trainable_weights,
    )


def _int8_config(*, dynamic_activation: bool) -> object:
    pytest.importorskip("torchao")
    try:
        from torchao.quantization import (
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
        )
    except ImportError as exc:
        # The int8 adapter targets the torchao>=0.17 version-2 Int8Tensor
        # workflow; skip (don't error) when the installed torchao predates
        # it — or a future release moves it.
        pytest.skip(f"torchao int8 API unavailable: {exc}")

    return Int8DynamicActivationInt8WeightConfig(version=2) if dynamic_activation else Int8WeightOnlyConfig(version=2)


def _int8_tensor_cls() -> type:
    pytest.importorskip("torchao")
    try:
        from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
            Int8Tensor,
        )
    except ImportError as exc:
        pytest.skip(f"torchao Int8Tensor unavailable: {exc}")
    return Int8Tensor


def _make_int8(
    *,
    rows: int = 32,
    cols: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    dynamic_activation: bool = False,
    weight: torch.Tensor | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    from torchao.quantization import quantize_

    cfg = _int8_config(dynamic_activation=dynamic_activation)
    layer = nn.Linear(cols, rows, bias=False).to(dtype)
    if weight is not None:
        with torch.no_grad():
            layer.weight.copy_(weight)
    layer = layer.to(device)
    quantize_(layer, cfg)
    return layer.weight.data


def _make_int8_pergroup(
    *,
    rows: int = 32,
    cols: int = 128,
    group_size: int = 64,
) -> torch.Tensor:
    pytest.importorskip("torchao")
    try:
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
        from torchao.quantization.granularity import PerGroup
    except ImportError as exc:
        pytest.skip(f"torchao per-group int8 API unavailable: {exc}")

    layer = nn.Linear(cols, rows, bias=False).to(torch.bfloat16)
    quantize_(layer, Int8WeightOnlyConfig(granularity=PerGroup(group_size), version=2))
    return layer.weight.data


class TestInt8Adapter:
    def test_matches_and_dispatches_int8_only(self) -> None:
        qt = _make_int8()
        assert Int8Adapter.matches(qt)
        assert not Int8Adapter.matches(torch.zeros(16, 16, dtype=torch.bfloat16))
        # Registry dispatch resolves Int8Tensor to this adapter (disjoint
        # from the other TorchAO structured adapters in the dispatch order).
        assert isinstance(select_adapter(qt), Int8Adapter)

    def test_pin_preserves_storage_and_metadata(self) -> None:
        int8_cls = _int8_tensor_cls()
        qt = _make_int8()
        pinned_param = PinnedParam(nn.Parameter(qt, requires_grad=False))

        pinned = pinned_param.make_cpu_param().data
        assert isinstance(pinned, int8_cls)
        assert pinned.qdata.is_pinned()
        assert pinned.scale.is_pinned()
        assert pinned.qdata.data_ptr() == pinned_param.pinned_state.storage[0].data_ptr()
        assert pinned.scale.data_ptr() == pinned_param.pinned_state.storage[1].data_ptr()
        if qt.zero_point is not None:
            assert pinned.zero_point is not None
            assert pinned.zero_point.is_pinned()
            assert pinned.zero_point.data_ptr() == pinned_param.pinned_state.storage[2].data_ptr()
        assert pinned.block_size == qt.block_size
        assert pinned.dtype == qt.dtype
        assert pinned_param.compute_dtype is torch.bfloat16
        assert torch.equal(pinned.dequantize(), qt.dequantize())

    def test_tensor_id_tracks_buffers(self) -> None:
        qt = _make_int8()
        key = tensor_id(qt)
        assert key[0] == "torchao-int8"
        assert key[1][0] == qt.qdata.device
        assert key[2][0] == qt.scale.device
        assert key == tensor_id(qt)
        assert key != tensor_id(_make_int8())

    def test_target_layout_ignores_tensor_id(self) -> None:
        p1 = nn.Parameter(_make_int8(), requires_grad=False)
        p2 = nn.Parameter(_make_int8(), requires_grad=False)

        assert _param_target_layout(p1) == _param_target_layout(p2)

    def test_target_layout_tracks_activation_quantization(self) -> None:
        with_activation = nn.Parameter(_make_int8(dynamic_activation=True), requires_grad=False)
        weight_only = nn.Parameter(_make_int8(dynamic_activation=False), requires_grad=False)

        assert _param_target_layout(with_activation) != _param_target_layout(weight_only)

    def test_no_cpu_round_trip_or_trainable_swap_capability(self) -> None:
        pinned_param = PinnedParam(
            nn.Parameter(_make_int8(), requires_grad=True),
        )
        state = pinned_param.allocate_gpu_storage(torch.device("cpu"))

        with pytest.raises(NotImplementedError, match="CPU round-trip"):
            pinned_param.copy_to_cpu(state)
        with pytest.raises(NotImplementedError, match="Parameter.data-swap"):
            pinned_param.validate_parameter_data_swap_target()

    @pytest.mark.parametrize("dynamic_activation", [False, True])
    def test_dequantize_requantize_preserves_representation(self, dynamic_activation: bool) -> None:
        int8_cls = _int8_tensor_cls()
        qt = _make_int8(rows=32, cols=16, dynamic_activation=dynamic_activation)
        dense = Int8Adapter.dequantize(qt)
        assert dense.dtype is torch.float32
        torch.testing.assert_close(dense, qt.dequantize().to(torch.float32))

        again = Int8Adapter.requantize(dense, like=qt)
        assert isinstance(again, int8_cls)
        assert again.block_size == qt.block_size
        assert again.dtype == qt.dtype
        assert again.qdata.dtype is torch.int8
        assert tuple(again.qdata.shape) == tuple(qt.qdata.shape)
        assert again.act_quant_kwargs == qt.act_quant_kwargs
        # int8's 256-level grid is too coarse for a bit-exact round trip
        # (boundary values flip ±1 bucket), but re-encoding stays within one
        # quantization step of the dense input it was given. Dequantize
        # directly to fp32: the default bf16 dequant rounds qdata*scale and
        # would occasionally push the error a hair past the bound.
        err = (again.dequantize(torch.float32) - dense).abs()
        assert err.max().item() <= again.scale.to(torch.float32).max().item()

    def test_requantize_rejects_shape_mismatch(self) -> None:
        qt = _make_int8(rows=32, cols=16)
        with pytest.raises(ValueError, match="Cannot requantize"):
            Int8Adapter.requantize(torch.randn(16, 32), like=qt)

    def test_requantize_recovers_per_group_granularity(self) -> None:
        # Int8WeightOnlyConfig(granularity=PerGroup(g)) gives block_size
        # [1, g] with g < in_features; requantize must recover PerGroup and
        # reproduce the partition rather than rejecting it as non-PerRow.
        int8_cls = _int8_tensor_cls()
        qt = _make_int8_pergroup(rows=32, cols=128, group_size=64)
        assert list(qt.block_size) == [1, 64]

        again = Int8Adapter.requantize(Int8Adapter.dequantize(qt), like=qt)
        assert isinstance(again, int8_cls)
        assert list(again.block_size) == [1, 64]
        assert tuple(again.scale.shape) == tuple(qt.scale.shape)

    def test_merge_lora_merges_per_group_int8_weight(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(128, 32, bias=False, dtype=torch.bfloat16)

        model = M()
        model.lin.weight.requires_grad = False
        model.lin.weight = nn.Parameter(
            _make_int8_pergroup(rows=32, cols=128, group_size=64),
            requires_grad=False,
        )
        original_qdata = model.lin.weight.data.qdata.clone()
        lora = LoRA.from_state_dict(
            state_dict={
                "lin.lora_A.weight": torch.randn(4, 128),
                "lin.lora_B.weight": torch.randn(32, 4),
            }
        )

        merged = merge_lora(model, [(lora, 1.0)])

        assert merged == 1
        assert list(model.lin.weight.data.block_size) == [1, 64]
        assert not torch.equal(model.lin.weight.data.qdata, original_qdata)

    def test_copy_into_preserves_absent_zero_point(self) -> None:
        # A symmetric int8 weight may carry zero_point=None, but
        # Int8Tensor.from_hp (used by requantize) always re-emits a zeros
        # zero_point. copy_into must fill only the slots the target has, not
        # assert on the recomputed zero_point the target lacks.
        int8_cls = _int8_tensor_cls()
        base = _make_int8(rows=32, cols=16)
        like = int8_cls(
            base.qdata,
            base.scale,
            list(base.block_size),
            base.dtype,
            zero_point=None,
        )
        assert like.zero_point is None

        dense = Int8Adapter.dequantize(like)
        dense.addmm_(torch.randn(32, 4), torch.randn(4, 16), alpha=0.5)
        new = Int8Adapter.requantize(dense, like=like)
        assert new.zero_point is not None  # from_hp always emits one

        Int8Adapter.copy_into(new, target=like)  # must not raise
        assert like.zero_point is None  # target representation preserved
        assert torch.equal(like.qdata, new.qdata)

    def test_lora_transform_requantizes_param_in_place(self) -> None:
        int8_cls = _int8_tensor_cls()
        rows, cols, rank = 32, 16, 2
        qt = _make_int8(rows=rows, cols=cols, dynamic_activation=False)
        param = nn.Parameter(qt, requires_grad=False)
        a = torch.randn(rank, cols)
        b = torch.randn(rows, rank)
        transform = LoRATransform([ScaledLoRAFactor(a, b, 0.5)])
        original_param = param
        original_qdata_ptr = param.data.qdata.data_ptr()

        # The merge path dequantizes, applies the delta, then requantizes;
        # mirror it exactly so the comparison is deterministic (not a lossy
        # round-trip property).
        expected_dense = qt.dequantize().to(torch.float32)
        expected_dense.addmm_(b.to(torch.float32), a.to(torch.float32), alpha=0.5)
        expected = Int8Adapter.requantize(expected_dense, like=qt)

        transform.apply(param)

        assert param is original_param
        assert param.data.qdata.data_ptr() == original_qdata_ptr
        assert isinstance(param.data, int8_cls)
        assert torch.equal(param.data.qdata, expected.qdata)
        assert torch.equal(param.data.scale, expected.scale)

    def test_merge_lora_merges_int8_weight(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)

        model = M()
        model.lin.weight.requires_grad = False
        model.lin.weight = nn.Parameter(
            _make_int8(rows=16, cols=16, dynamic_activation=False),
            requires_grad=False,
        )
        # copy_into mutates the weight's storage in place, so snapshot the
        # original int8 bytes rather than holding a tensor ref.
        original_qdata = model.lin.weight.data.qdata.clone()
        lora = LoRA.from_state_dict(
            state_dict={
                "lin.lora_A.weight": torch.randn(4, 16),
                "lin.lora_B.weight": torch.randn(16, 4),
            }
        )

        merged = merge_lora(model, [(lora, 1.0)])

        assert merged == 1
        assert not torch.equal(model.lin.weight.data.qdata, original_qdata)

    def test_reconstructed_cpu_forward_matches(self) -> None:
        # int8 matmul runs on CPU, so reconstruction correctness is checked
        # without a GPU: the rebuilt wrapper must produce the same output.
        int8_cls = _int8_tensor_cls()
        weight = torch.randn(32, 16, dtype=torch.bfloat16)
        qt = _make_int8(rows=32, cols=16, weight=weight)
        x = torch.randn(4, 16, dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, qt)

        pinned_param = PinnedParam(nn.Parameter(qt, requires_grad=False))
        reconstructed = pinned_param.make_cpu_param().data
        assert isinstance(reconstructed, int8_cls)
        out = torch.nn.functional.linear(x, reconstructed)
        torch.testing.assert_close(out, ref)

    @CUDA
    def test_allocate_copy_make_gpu_param_preserves_wrapper(self) -> None:
        int8_cls = _int8_tensor_cls()
        qt = _make_int8()
        pinned_param = PinnedParam(nn.Parameter(qt, requires_grad=False))

        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()
        pinned = pinned_param.make_cpu_param().data

        assert isinstance(gpu_param.data, int8_cls)
        assert gpu_param.data.qdata.is_cuda
        assert gpu_param.data.scale.is_cuda
        assert gpu_param.data.block_size == pinned.block_size
        assert gpu_param.data.dtype == pinned.dtype
        assert torch.equal(gpu_param.data.qdata.cpu(), pinned.qdata)
        assert torch.equal(gpu_param.data.scale.cpu(), pinned.scale)
        if pinned.zero_point is not None:
            assert gpu_param.data.zero_point is not None
            assert torch.equal(gpu_param.data.zero_point.cpu(), pinned.zero_point)

    @CUDA
    @pytest.mark.parametrize("dynamic_activation", [False, True])
    def test_model_offloader_cuda_forward(self, dynamic_activation: bool) -> None:
        layer = nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
        layer.weight.requires_grad = False
        layer.weight = nn.Parameter(
            _make_int8(rows=128, cols=64, dynamic_activation=dynamic_activation),
            requires_grad=False,
        )
        strategy = _make_model_offloader(layer)

        try:
            x = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
            with activated_model(strategy, "cuda") as active:
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (128, 128)
            assert y.dtype is torch.bfloat16
        finally:
            strategy.deactivate()

    @CUDA
    def test_model_offloader_routed_lora_on_int8(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        nn.Linear(128, 128, bias=False, dtype=torch.bfloat16),
                        nn.Linear(128, 128, bias=False, dtype=torch.bfloat16),
                    ]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = block(x)
                return x

        model = M()
        for block in model.blocks:
            block.weight.requires_grad = False
            block.weight = nn.Parameter(
                _make_int8(rows=128, cols=128, dynamic_activation=True),
                requires_grad=False,
            )
        offloader = _make_model_offloader(
            model,
            blocks_attr=["blocks"],
        )
        lora = LoRA.from_state_dict(
            state_dict={
                "blocks.0.lora_A.weight": torch.randn(4, 128),
                "blocks.0.lora_B.weight": torch.randn(128, 4),
            }
        )
        try:
            x = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
            with activated_model(offloader,
                "cuda",
                loras=[lora],
                lora_strengths=[0.25],
                lora_mode="routed",
                stream_config=StreamConfig(num_resident_blocks=1, num_prefetch_blocks=0),
            ) as active:
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (128, 128)
            assert y.dtype is torch.bfloat16
        finally:
            offloader.deactivate()

    @CUDA
    def test_streamed_int8_merge_requantizes_on_activate(self) -> None:
        int8_cls = _int8_tensor_cls()

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        nn.Linear(32, 32, bias=False, dtype=torch.bfloat16),
                        nn.Linear(32, 32, bias=False, dtype=torch.bfloat16),
                    ]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = block(x)
                return x

        model = M()
        for block in model.blocks:
            block.weight.requires_grad = False
            block.weight = nn.Parameter(
                _make_int8(rows=32, cols=32, dynamic_activation=False),
                requires_grad=False,
            )
        qt = model.blocks[0].weight.data
        rank = 4
        a = torch.randn(rank, 32)
        b = torch.randn(32, rank)
        lora = LoRA.from_state_dict(
            state_dict={
                "blocks.0.lora_A.weight": a,
                "blocks.0.lora_B.weight": b,
            }
        )
        # Reference on CUDA, matching the device the offloader merges on: the
        # offloader merges into a byte-identical GPU copy of the weight.
        qt_cuda = qt.cuda()
        expected_dense = qt_cuda.dequantize().to(torch.float32)
        expected_dense.addmm_(b.cuda().to(torch.float32), a.cuda().to(torch.float32), alpha=0.5)
        expected = Int8Adapter.requantize(expected_dense, like=qt_cuda)

        offloader = _make_model_offloader(
            model,
            blocks_attr=["blocks"],
        )
        try:
            x = torch.randn(8, 32, dtype=torch.bfloat16, device="cuda")
            with activated_model(offloader,
                "cuda",
                loras=[lora],
                lora_strengths=[0.5],
                lora_mode="merge",
                stream_config=StreamConfig(num_resident_blocks=1, num_prefetch_blocks=0),
            ) as active:
                merged = active.blocks[0].weight.data
                assert isinstance(merged, int8_cls)
                torch.testing.assert_close(
                    merged.dequantize().to(torch.float32),
                    expected.dequantize().to(torch.float32),
                )
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (8, 32)
        finally:
            offloader.deactivate()
