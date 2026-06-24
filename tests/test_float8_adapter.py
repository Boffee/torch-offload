"""Tests for TorchAO scaled-fp8 (``Float8Tensor``) adapter integration."""

from __future__ import annotations


import pytest
import torch
from torch import nn

from torch_offload import LoRA, LoRATransform, ModelOffloader, ModelOffloaderStore, ScaledLoRAFactor, merge_lora
from torch_offload.float8_adapter import Float8Adapter
from torch_offload.pinned_param import PinnedParam
from torch_offload.streamed_component import _param_target_layout
from torch_offload.tensor_adapter_registry import tensor_id

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_model_offloader(
    model: nn.Module,
    *,
    blocks_attr: list[str] = [],
    num_resident_blocks: int = 1,
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


def _float8_modules():
    pytest.importorskip("torchao")
    try:
        from torchao.quantization.granularity import PerRow, PerTensor
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            Float8Tensor,
            QuantizeTensorToFloat8Kwargs,
        )
    except ImportError as exc:
        # The float8 adapter targets the torchao>=0.17 Float8Tensor workflow;
        # skip (don't error) when the installed torchao predates it — or a
        # future release moves it — matching the importorskip above.
        pytest.skip(f"torchao float8 API unavailable: {exc}")

    return Float8Tensor, QuantizeTensorToFloat8Kwargs, PerRow, PerTensor


def _mm_config() -> object:
    # The quantize_(...) workflow always sets mm_config on weights; the
    # scaled-mm forward path asserts it is present. Match that here.
    from torchao.float8.inference import Float8MMConfig

    return Float8MMConfig(use_fast_accum=True)


def _make_float8(
    *,
    rows: int = 16,
    cols: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    per_tensor: bool = False,
    dynamic_activation: bool = True,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    float8_tensor_cls, kwargs_cls, per_row_cls, per_tensor_cls = _float8_modules()
    granularity = per_tensor_cls() if per_tensor else per_row_cls()
    act_quant_kwargs = (
        kwargs_cls(granularity=granularity) if dynamic_activation else None
    )
    if weight is None:
        weight = torch.randn(rows, cols, dtype=dtype)
    return float8_tensor_cls.from_hp(
        weight,
        granularity=granularity,
        mm_config=_mm_config(),
        act_quant_kwargs=act_quant_kwargs,
    )


class TestFloat8Adapter:
    def test_matches_float8_only(self) -> None:
        f8 = _make_float8()
        assert Float8Adapter.matches(f8)
        assert not Float8Adapter.matches(torch.zeros(16, 16, dtype=torch.bfloat16))

    def test_pin_preserves_storage_and_metadata(self) -> None:
        float8_tensor_cls, _, _, _ = _float8_modules()
        f8 = _make_float8()
        p = nn.Parameter(f8, requires_grad=False)
        pinned_param = PinnedParam(p)

        pinned = pinned_param.make_cpu_param().data
        assert isinstance(pinned, float8_tensor_cls)
        assert pinned.qdata.is_pinned()
        assert pinned.scale.is_pinned()
        assert pinned.qdata.data_ptr() == pinned_param.pinned_state.storage[0].data_ptr()
        assert pinned.scale.data_ptr() == pinned_param.pinned_state.storage[1].data_ptr()
        assert pinned.block_size == f8.block_size
        assert pinned.mm_config == f8.mm_config
        assert pinned.kernel_preference == f8.kernel_preference
        assert pinned.act_quant_kwargs == f8.act_quant_kwargs
        assert pinned.dtype == f8.dtype
        assert pinned_param.compute_dtype is torch.bfloat16
        assert torch.equal(pinned.dequantize(), f8.dequantize())

    def test_tensor_id_tracks_both_buffers(self) -> None:
        f8 = _make_float8()
        key = tensor_id(f8)
        assert key[0] == "torchao-float8"
        assert key[1][0] == f8.qdata.device
        assert key[2][0] == f8.scale.device
        assert key == tensor_id(f8)
        assert key != tensor_id(_make_float8())

    def test_target_layout_ignores_tensor_id(self) -> None:
        p1 = nn.Parameter(_make_float8(), requires_grad=False)
        p2 = nn.Parameter(_make_float8(), requires_grad=False)

        assert _param_target_layout(p1) == _param_target_layout(p2)

    def test_target_layout_tracks_granularity(self) -> None:
        per_row = nn.Parameter(
            _make_float8(per_tensor=False), requires_grad=False
        )
        per_tensor = nn.Parameter(
            _make_float8(per_tensor=True), requires_grad=False
        )

        assert _param_target_layout(per_row) != _param_target_layout(per_tensor)

    def test_target_layout_tracks_activation_quantization(self) -> None:
        with_activation = nn.Parameter(
            _make_float8(dynamic_activation=True), requires_grad=False
        )
        weight_only = nn.Parameter(
            _make_float8(dynamic_activation=False), requires_grad=False
        )

        assert _param_target_layout(with_activation) != _param_target_layout(
            weight_only
        )

    def test_cpu_round_trip_restores_pinned_bytes(self) -> None:
        pinned_param = PinnedParam(
            nn.Parameter(_make_float8(), requires_grad=False),
        )
        state = pinned_param.allocate_gpu_storage(torch.device("cpu"))
        pinned_param.copy_to_gpu(state)

        original_qdata = pinned_param.pinned_state.storage[0].view(torch.uint8).clone()
        original_scale = pinned_param.pinned_state.storage[1].clone()
        pinned_param.pinned_state.storage[0].view(torch.uint8).zero_()
        pinned_param.pinned_state.storage[1].zero_()
        pinned_param.copy_to_cpu(state)

        assert torch.equal(
            pinned_param.pinned_state.storage[0].view(torch.uint8), original_qdata
        )
        assert torch.equal(pinned_param.pinned_state.storage[1], original_scale)

    def test_no_trainable_swap_capability(self) -> None:
        pinned_param = PinnedParam(
            nn.Parameter(_make_float8(), requires_grad=True),
        )

        with pytest.raises(NotImplementedError, match="Parameter.data-swap"):
            pinned_param.validate_parameter_data_swap_target()

    @pytest.mark.parametrize("per_tensor", [False, True])
    def test_dequantize_requantize_preserves_representation(
        self, per_tensor: bool
    ) -> None:
        f8 = _make_float8(per_tensor=per_tensor)
        dense = Float8Adapter.dequantize(f8)
        assert dense.dtype is torch.float32
        torch.testing.assert_close(dense, f8.dequantize().to(torch.float32))

        again = Float8Adapter.requantize(dense, like=f8)
        assert again.block_size == f8.block_size
        assert again.qdata.dtype == f8.qdata.dtype
        assert again.dtype == f8.dtype
        assert again.kernel_preference == f8.kernel_preference
        assert again.mm_config == f8.mm_config
        assert again.act_quant_kwargs == f8.act_quant_kwargs
        assert torch.equal(
            again.qdata.view(torch.uint8), f8.qdata.view(torch.uint8)
        )
        assert torch.equal(again.scale, f8.scale)

    def test_requantize_rejects_shape_mismatch(self) -> None:
        f8 = _make_float8(rows=4, cols=8)
        with pytest.raises(ValueError, match="Cannot requantize"):
            Float8Adapter.requantize(torch.randn(8, 4), like=f8)

    def test_requantize_zero_row_does_not_nan(self) -> None:
        # torchao's from_hp computes scale = amax / fp8_max with no eps
        # floor, so an all-zero row (per-row scaling) gets scale 0 and
        # qdata 0/0 = NaN. requantize must repair that to an exact zero row
        # while leaving the other rows intact.
        f8 = _make_float8(rows=16, cols=64, dynamic_activation=False)
        dense = Float8Adapter.dequantize(f8)
        dense[3] = 0  # one fully cancelled row

        again = Float8Adapter.requantize(dense, like=f8)
        deq = again.dequantize().to(torch.float32)
        assert not torch.isnan(deq).any()
        assert torch.count_nonzero(deq[3]).item() == 0
        assert torch.count_nonzero(deq[[0, 1, 2, 4]]).item() > 0

    def test_requantize_all_zero_does_not_nan(self) -> None:
        # Per-tensor scaling: a fully cancelled weight gives a scalar scale
        # of 0; the repair must still yield a clean all-zero tensor.
        f8 = _make_float8(rows=16, cols=16, per_tensor=True)
        again = Float8Adapter.requantize(
            torch.zeros(16, 16, dtype=torch.float32), like=f8
        )
        deq = again.dequantize().to(torch.float32)
        assert not torch.isnan(deq).any()
        assert torch.count_nonzero(deq).item() == 0

    def test_lora_transform_requantizes_param_in_place(self) -> None:
        float8_tensor_cls, _, per_row_cls, _ = _float8_modules()
        rows, cols, rank = 4, 8, 2
        f8 = _make_float8(rows=rows, cols=cols, dynamic_activation=False)
        param = nn.Parameter(f8, requires_grad=False)
        a = torch.randn(rank, cols)
        b = torch.randn(rows, rank)
        transform = LoRATransform([ScaledLoRAFactor(a, b, 0.5)])
        original_param = param
        original_qdata_ptr = param.data.qdata.data_ptr()

        expected_dense = f8.dequantize().to(torch.float32)
        expected_dense.addmm_(b.to(torch.float32), a.to(torch.float32), alpha=0.5)
        expected = float8_tensor_cls.from_hp(
            expected_dense.to(f8.dtype), granularity=per_row_cls(),
        )

        transform.apply(param)

        assert param is original_param
        assert param.data.qdata.data_ptr() == original_qdata_ptr
        assert isinstance(param.data, float8_tensor_cls)
        assert torch.equal(
            param.data.qdata.view(torch.uint8),
            expected.qdata.view(torch.uint8),
        )
        assert torch.equal(param.data.scale, expected.scale)

    def test_merge_lora_merges_float8_weight(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)

        model = M()
        model.lin.weight.requires_grad = False
        model.lin.weight = nn.Parameter(
            _make_float8(dynamic_activation=False), requires_grad=False
        )
        # copy_into mutates the weight's storage in place, so snapshot
        # the original packed bytes rather than holding a tensor ref.
        original_qdata = model.lin.weight.data.qdata.view(torch.uint8).clone()
        lora = LoRA(
            state_dict={
                "lin.lora_A.weight": torch.randn(4, 16),
                "lin.lora_B.weight": torch.randn(16, 4),
            }
        )

        merged = merge_lora(model, [(lora, 1.0)])

        assert merged == 1
        assert not torch.equal(
            model.lin.weight.data.qdata.view(torch.uint8), original_qdata
        )

    @CUDA
    def test_allocate_copy_make_gpu_param_preserves_wrapper(self) -> None:
        float8_tensor_cls, _, _, _ = _float8_modules()
        pinned_param = PinnedParam(
            nn.Parameter(_make_float8(), requires_grad=False),
        )

        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()
        pinned = pinned_param.make_cpu_param().data

        assert isinstance(gpu_param.data, float8_tensor_cls)
        assert gpu_param.data.qdata.is_cuda
        assert gpu_param.data.scale.is_cuda
        assert gpu_param.data.block_size == pinned.block_size
        assert gpu_param.data.kernel_preference == pinned.kernel_preference
        assert gpu_param.data.act_quant_kwargs == pinned.act_quant_kwargs
        assert gpu_param.data.dtype == pinned.dtype
        assert torch.equal(
            gpu_param.data.qdata.view(torch.uint8).cpu(),
            pinned.qdata.view(torch.uint8),
        )
        assert torch.equal(gpu_param.data.scale.cpu(), pinned.scale)

    @CUDA
    def test_model_offloader_cuda_forward_dynamic_float8(self) -> None:
        layer = nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
        layer.weight.requires_grad = False
        weight = layer.weight.detach().contiguous()
        layer.weight = nn.Parameter(
            _make_float8(weight=weight, dynamic_activation=True),
            requires_grad=False,
        )
        strategy = _make_model_offloader(layer)

        try:
            x = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
            with strategy.use("cuda") as active:
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (128, 128)
            assert y.dtype is torch.bfloat16
        finally:
            strategy.deactivate()

    @CUDA
    def test_streamed_float8_merge_requantizes_on_activate(self) -> None:
        float8_tensor_cls, _, per_row_cls, _ = _float8_modules()

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        nn.Linear(16, 16, bias=False, dtype=torch.bfloat16),
                        nn.Linear(16, 16, bias=False, dtype=torch.bfloat16),
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
                _make_float8(dynamic_activation=False),
                requires_grad=False,
            )
        f8 = model.blocks[0].weight.data
        rank = 4
        a = torch.randn(rank, 16)
        b = torch.randn(16, rank)
        lora = LoRA(
            state_dict={
                "blocks.0.lora_A.weight": a,
                "blocks.0.lora_B.weight": b,
            }
        )
        # Compute the reference on CUDA, matching the device the offloader
        # merges on. A CPU reference flips a couple of float8 boundary elements
        # relative to the CUDA merge (CPU vs CUDA round-to-nearest at bucket
        # edges), making the tight tolerance RNG/CUDA-state sensitive.
        f8_cuda = f8.cuda()
        expected_dense = f8_cuda.dequantize().to(torch.float32)
        expected_dense.addmm_(
            b.cuda().to(torch.float32), a.cuda().to(torch.float32), alpha=0.5
        )
        expected = float8_tensor_cls.from_hp(
            expected_dense.to(f8.dtype), granularity=per_row_cls(),
        )

        offloader = _make_model_offloader(
            model,
            blocks_attr=["blocks"],
            num_resident_blocks=1,
            num_prefetch_blocks=0,
        )
        offloader.set_loras([lora], strengths=[0.5], mode="merge")

        try:
            x = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
            with offloader.use("cuda") as active:
                merged = active.blocks[0].weight.data
                assert isinstance(merged, float8_tensor_cls)
                torch.testing.assert_close(
                    merged.dequantize().to(torch.float32),
                    expected.dequantize().to(torch.float32),
                )
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (8, 16)
        finally:
            offloader.deactivate()
