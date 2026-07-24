"""Tests for TorchAO static-activation ``PrototypeFloat8Tensor`` support."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torch_offload._triton_static_float8 import merge_static_float8_lora
from torch_offload import (
    BlockCompileConfig,
    LoRA,
    LoRATransform,
    ModelOffloader,
    ScaledLoRAFactor,
    StreamConfig,
    merge_lora,
)
from torch_offload.float8_adapter import Float8Adapter
from torch_offload.pinned_param import PinnedParam
from torch_offload.static_float8_adapter import StaticFloat8Adapter
from torch_offload.streamed_component import _param_target_layout
from torch_offload.tensor_adapter_registry import select_adapter, tensor_id
from tests.conftest import activated_model

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _static_float8_modules():
    pytest.importorskip("torchao")
    try:
        from torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor import (
            PrototypeFloat8Tensor,
        )
        from torchao.quantization.granularity import PerRow, PerTensor
        from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
            QuantizeTensorToFloat8Kwargs,
        )
    except ImportError as exc:
        pytest.skip(f"torchao static float8 prototype API unavailable: {exc}")
    return PrototypeFloat8Tensor, QuantizeTensorToFloat8Kwargs, PerRow, PerTensor


def _mm_config() -> object:
    from torchao.float8.inference import Float8MMConfig

    return Float8MMConfig(use_fast_accum=True)


def _make_static_float8(
    *,
    rows: int = 16,
    cols: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    float8_dtype: torch.dtype = torch.float8_e4m3fn,
    act_scale_shape: tuple[int, ...] = (),
    act_scale_value: float = 0.02,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    prototype_cls, kwargs_cls, _, per_tensor_cls = _static_float8_modules()
    if weight is None:
        weight = torch.randn(rows, cols, dtype=dtype)
    mm_config = _mm_config()
    return prototype_cls.from_hp(
        weight,
        float8_dtype=float8_dtype,
        granularity=per_tensor_cls(),
        mm_config=mm_config,
        act_quant_kwargs=kwargs_cls(
            granularity=per_tensor_cls(),
            mm_config=mm_config,
        ),
        act_quant_scale=torch.full(
            act_scale_shape, act_scale_value, dtype=torch.float32
        ),
    )


def _make_prototype_without_static_scale() -> torch.Tensor:
    prototype_cls, _, _, per_tensor_cls = _static_float8_modules()
    return prototype_cls.from_hp(
        torch.randn(16, 16, dtype=torch.bfloat16),
        granularity=per_tensor_cls(),
        mm_config=_mm_config(),
    )


def _replace_act_scale(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    prototype_cls, _, _, _ = _static_float8_modules()
    return prototype_cls(
        t.qdata,
        t.scale,
        act_quant_scale=scale,
        block_size=list(t.block_size),
        mm_config=t.mm_config,
        act_quant_kwargs=t.act_quant_kwargs,
        kernel_preference=t.kernel_preference,
        dtype=t.dtype,
    )


def _make_model_offloader(
    model: nn.Module,
    *,
    blocks_attr: list[str] = [],
    block_compile: BlockCompileConfig | None = None,
) -> ModelOffloader:
    return ModelOffloader.from_module(
        model,
        blocks_attr=blocks_attr,
        block_compile=block_compile,
    )


class TestStaticFloat8Adapter:
    def test_matches_only_calibrated_prototype_tensor(self) -> None:
        static = _make_static_float8()
        prototype_without_scale = _make_prototype_without_static_scale()

        assert StaticFloat8Adapter.matches(static)
        assert not Float8Adapter.matches(static)
        assert not StaticFloat8Adapter.matches(prototype_without_scale)
        assert not StaticFloat8Adapter.matches(torch.zeros(16, 16))
        assert isinstance(select_adapter(static), StaticFloat8Adapter)

    def test_rejects_non_per_tensor_weight_layout(self) -> None:
        prototype_cls, kwargs_cls, per_row_cls, per_tensor_cls = (
            _static_float8_modules()
        )
        mm_config = _mm_config()
        per_row_weight = prototype_cls.from_hp(
            torch.randn(16, 16, dtype=torch.bfloat16),
            granularity=per_row_cls(),
            mm_config=mm_config,
            act_quant_kwargs=kwargs_cls(
                granularity=per_tensor_cls(),
                mm_config=mm_config,
            ),
            act_quant_scale=torch.tensor(0.02),
        )

        with pytest.raises(ValueError, match="per-tensor FP8 weight layout"):
            StaticFloat8Adapter.matches(per_row_weight)

    def test_pin_preserves_three_storage_tensors_and_metadata(self) -> None:
        prototype_cls, _, _, _ = _static_float8_modules()
        f8 = _make_static_float8(act_scale_shape=(1, 1, 1))
        pinned_param = PinnedParam(nn.Parameter(f8, requires_grad=False))
        pinned = pinned_param.make_cpu_param().data

        assert isinstance(pinned, prototype_cls)
        assert len(pinned_param.pinned_state.storage) == 3
        assert pinned.qdata.is_pinned()
        assert pinned.scale.is_pinned()
        assert pinned.act_quant_scale.is_pinned()
        assert pinned.qdata.data_ptr() == pinned_param.pinned_state.storage[0].data_ptr()
        assert pinned.scale.data_ptr() == pinned_param.pinned_state.storage[1].data_ptr()
        assert (
            pinned.act_quant_scale.data_ptr()
            == pinned_param.pinned_state.storage[2].data_ptr()
        )
        assert pinned.block_size == f8.block_size
        assert pinned.mm_config == f8.mm_config
        assert pinned.kernel_preference == f8.kernel_preference
        assert pinned.act_quant_kwargs == f8.act_quant_kwargs
        assert pinned.dtype == f8.dtype
        assert pinned_param.compute_dtype is torch.bfloat16
        assert torch.equal(pinned.act_quant_scale, f8.act_quant_scale)
        assert torch.equal(pinned.dequantize(), f8.dequantize())

    def test_cache_bytes_include_calibrated_scale(self) -> None:
        pinned_param = PinnedParam(
            nn.Parameter(_make_static_float8(), requires_grad=False)
        )
        expected = sum(t.nbytes for t in pinned_param.pinned_state.storage)

        assert expected == 16 * 16 + 4 + 4
        assert pinned_param.cache_bytes == expected

    def test_tensor_id_tracks_activation_scale_identity(self) -> None:
        f8 = _make_static_float8()
        same_weight_new_scale = _replace_act_scale(
            f8, f8.act_quant_scale.detach().clone()
        )
        key = tensor_id(f8)

        assert key[0] == "torchao-static-float8"
        assert key[1][0] == f8.qdata.device
        assert key[2][0] == f8.scale.device
        assert key[3][0] == f8.act_quant_scale.device
        assert key == tensor_id(f8)
        assert key != tensor_id(same_weight_new_scale)

    def test_target_layout_ignores_storage_identity(self) -> None:
        p1 = nn.Parameter(
            _make_static_float8(act_scale_value=0.02), requires_grad=False
        )
        p2 = nn.Parameter(
            _make_static_float8(act_scale_value=0.04), requires_grad=False
        )

        assert _param_target_layout(p1) == _param_target_layout(p2)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_reports_logical_compute_dtype(self, dtype: torch.dtype) -> None:
        f8 = _make_static_float8(dtype=dtype)

        assert StaticFloat8Adapter.compute_dtype(f8) is dtype

    def test_target_layout_tracks_activation_scale_layout(self) -> None:
        scalar = nn.Parameter(
            _make_static_float8(act_scale_shape=()), requires_grad=False
        )
        rank_two = nn.Parameter(
            _make_static_float8(act_scale_shape=(1, 1)), requires_grad=False
        )

        assert _param_target_layout(scalar) != _param_target_layout(rank_two)

    def test_meta_layout_inspection_keeps_static_representation(self) -> None:
        f8 = _make_static_float8()
        meta = _replace_act_scale(
            f8,
            f8.act_quant_scale.to(device="meta"),
        )
        prototype_cls, _, _, _ = _static_float8_modules()
        meta = prototype_cls(
            f8.qdata.to(device="meta"),
            f8.scale.to(device="meta"),
            act_quant_scale=meta.act_quant_scale,
            block_size=list(f8.block_size),
            mm_config=f8.mm_config,
            act_quant_kwargs=f8.act_quant_kwargs,
            kernel_preference=f8.kernel_preference,
            dtype=f8.dtype,
        )

        assert StaticFloat8Adapter.matches(meta)
        signature = StaticFloat8Adapter.layout_signature(meta)
        assert signature[-1][0] == "act_quant_scale"
        assert signature[-1][1] == ((), torch.float32, ())

    def test_cpu_round_trip_restores_all_three_storage_tensors(self) -> None:
        pinned_param = PinnedParam(
            nn.Parameter(_make_static_float8(), requires_grad=False)
        )
        state = pinned_param.allocate_gpu_storage(torch.device("cpu"))
        pinned_param.copy_to_gpu(state)
        originals = tuple(t.clone() for t in state.storage)

        for tensor in pinned_param.pinned_state.storage:
            tensor.zero_()
        pinned_param.copy_to_cpu(state)

        for restored, original in zip(
            pinned_param.pinned_state.storage, originals, strict=True
        ):
            assert torch.equal(
                restored.reshape(-1).view(torch.uint8),
                original.reshape(-1).view(torch.uint8),
            )

    def test_no_trainable_swap_capability(self) -> None:
        pinned_param = PinnedParam(
            nn.Parameter(_make_static_float8(), requires_grad=True)
        )
        with pytest.raises(NotImplementedError, match="Parameter.data-swap"):
            pinned_param.validate_parameter_data_swap_target()

    def test_dequantize_requantize_preserves_static_representation(self) -> None:
        prototype_cls, _, _, _ = _static_float8_modules()
        f8 = _make_static_float8(act_scale_shape=(1, 1, 1))
        dense = StaticFloat8Adapter.dequantize(f8)

        assert dense.dtype is f8.dtype
        torch.testing.assert_close(dense, f8.dequantize())
        again = StaticFloat8Adapter.requantize(dense, like=f8)
        assert isinstance(again, prototype_cls)
        assert again.block_size == f8.block_size
        assert again.qdata.dtype == f8.qdata.dtype
        assert again.dtype == f8.dtype
        assert again.mm_config == f8.mm_config
        assert again.kernel_preference == f8.kernel_preference
        assert again.act_quant_kwargs == f8.act_quant_kwargs
        assert again.act_quant_scale is f8.act_quant_scale
        assert torch.equal(again.qdata.view(torch.uint8), f8.qdata.view(torch.uint8))
        assert torch.equal(again.scale, f8.scale)
        assert torch.equal(again.act_quant_scale, f8.act_quant_scale)

    def test_requantize_rejects_shape_mismatch(self) -> None:
        f8 = _make_static_float8(rows=4, cols=8)
        with pytest.raises(ValueError, match="Cannot requantize"):
            StaticFloat8Adapter.requantize(torch.randn(8, 4), like=f8)

    def test_requantize_all_zero_does_not_nan(self) -> None:
        f8 = _make_static_float8()
        again = StaticFloat8Adapter.requantize(
            torch.zeros(16, 16, dtype=torch.float32), like=f8
        )

        dequantized = again.dequantize().to(torch.float32)
        assert not torch.isnan(dequantized).any()
        assert torch.count_nonzero(dequantized).item() == 0
        assert torch.equal(again.act_quant_scale, f8.act_quant_scale)

    @CUDA
    @pytest.mark.parametrize(
        ("dtype", "float8_dtype"),
        [
            (torch.bfloat16, torch.float8_e4m3fn),
            (torch.bfloat16, torch.float8_e5m2),
            (torch.float16, torch.float8_e4m3fn),
            (torch.float16, torch.float8_e5m2),
            (torch.float32, torch.float8_e4m3fn),
            (torch.float32, torch.float8_e5m2),
        ],
    )
    def test_triton_lora_merge_matches_eager_round_trip(
        self,
        dtype: torch.dtype,
        float8_dtype: torch.dtype,
    ) -> None:
        rows, cols, rank = 70, 130, 7
        f8 = _make_static_float8(
            rows=rows,
            cols=cols,
            dtype=dtype,
            float8_dtype=float8_dtype,
        ).cuda()
        a = torch.randn(rank, cols, device="cuda", dtype=dtype)
        b = torch.randn(rows, rank, device="cuda", dtype=dtype)
        strength = 0.25

        dense = StaticFloat8Adapter.dequantize(f8)
        dense.addmm_(b, a, alpha=strength)
        expected = StaticFloat8Adapter.requantize(dense, like=f8)
        qdata, scale = merge_static_float8_lora(
            f8.qdata,
            f8.scale,
            b,
            a,
            strength,
        )

        torch.testing.assert_close(
            scale,
            expected.scale,
            rtol=0.02,
            atol=0,
        )
        torch.testing.assert_close(
            qdata.to(torch.float32) * scale,
            expected.dequantize().to(torch.float32),
            rtol=0.3 if float8_dtype is torch.float8_e5m2 else 0.13,
            atol=0.15 if float8_dtype is torch.float8_e5m2 else 0.05,
        )

    @CUDA
    def test_triton_lora_merge_encodes_zero_weight_safely(self) -> None:
        f8 = _make_static_float8().cuda()
        f8.qdata.zero_()
        f8.scale.fill_(torch.finfo(torch.float32).eps)
        a = torch.zeros(4, 16, device="cuda", dtype=f8.dtype)
        b = torch.zeros(16, 4, device="cuda", dtype=f8.dtype)

        qdata, scale = merge_static_float8_lora(
            f8.qdata,
            f8.scale,
            b,
            a,
            1.0,
        )

        dequantized = qdata.to(torch.float32) * scale
        assert torch.isfinite(dequantized).all()
        assert torch.count_nonzero(dequantized).item() == 0
        assert torch.count_nonzero(scale).item() == 1

    def test_copy_into_does_not_copy_activation_scale(self) -> None:
        target = _make_static_float8()
        dense = StaticFloat8Adapter.dequantize(target)
        dense.addmm_(
            torch.randn(16, 4, dtype=dense.dtype),
            torch.randn(4, 16, dtype=dense.dtype),
            alpha=0.5,
        )
        requantized = StaticFloat8Adapter.requantize(dense, like=target)
        src = _replace_act_scale(requantized, torch.tensor(123.0))
        original_scale = target.act_quant_scale.clone()
        original_scale_ptr = target.act_quant_scale.data_ptr()

        StaticFloat8Adapter.copy_into(src, target=target)

        assert target.act_quant_scale.data_ptr() == original_scale_ptr
        assert torch.equal(target.act_quant_scale, original_scale)
        assert torch.equal(target.qdata.view(torch.uint8), src.qdata.view(torch.uint8))
        assert torch.equal(target.scale, src.scale)

    def test_lora_transform_requantizes_weight_but_preserves_calibration(self) -> None:
        rows, cols, rank = 16, 16, 4
        f8 = _make_static_float8(rows=rows, cols=cols)
        param = nn.Parameter(f8, requires_grad=False)
        a = torch.randn(rank, cols)
        b = torch.randn(rows, rank)
        transform = LoRATransform(
            [ScaledLoRAFactor.from_tensors(a, b, 0.5)]
        )
        original_qdata_ptr = param.data.qdata.data_ptr()
        original_act_scale_ptr = param.data.act_quant_scale.data_ptr()
        original_act_scale = param.data.act_quant_scale.clone()
        expected_dense = StaticFloat8Adapter.dequantize(f8)
        expected_dense.addmm_(
            b.to(expected_dense.dtype),
            a.to(expected_dense.dtype),
            alpha=0.5,
        )
        expected = StaticFloat8Adapter.requantize(expected_dense, like=f8)

        transform.apply(param)

        assert param.data.qdata.data_ptr() == original_qdata_ptr
        assert param.data.act_quant_scale.data_ptr() == original_act_scale_ptr
        assert torch.equal(param.data.qdata.view(torch.uint8), expected.qdata.view(torch.uint8))
        assert torch.equal(param.data.scale, expected.scale)
        assert torch.equal(param.data.act_quant_scale, original_act_scale)

    @CUDA
    def test_triton_lora_transform_uses_raw_storage_and_preserves_wrapper(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rows, cols = 64, 64
        f8 = _make_static_float8(rows=rows, cols=cols).cuda()
        param = nn.Parameter(f8, requires_grad=False)
        factor_inputs = [
            (
                torch.randn(4, cols),
                torch.randn(rows, 4),
                0.5,
            ),
            (
                torch.randn(2, cols),
                torch.randn(rows, 2),
                -0.25,
            ),
        ]
        factors = [
            ScaledLoRAFactor.from_tensors(a, b, strength)
            for a, b, strength in factor_inputs
        ]
        dense = StaticFloat8Adapter.dequantize(f8)
        packed_a = torch.cat(
            [
                a.to(device="cuda", dtype=dense.dtype).mul_(strength)
                for a, _b, strength in factor_inputs
            ],
            dim=0,
        )
        packed_b = torch.cat(
            [
                b.to(device="cuda", dtype=dense.dtype)
                for _a, b, _strength in factor_inputs
            ],
            dim=1,
        )
        dense.addmm_(packed_b, packed_a)
        expected = StaticFloat8Adapter.requantize(dense, like=f8)
        qdata_ptr = f8.qdata.data_ptr()
        scale_ptr = f8.scale.data_ptr()
        act_scale_ptr = f8.act_quant_scale.data_ptr()

        def fail_dequantize(_tensor: torch.Tensor) -> torch.Tensor:
            raise AssertionError(
                "Triton merge must not build the generic dense wrapper path"
            )

        monkeypatch.setattr(
            StaticFloat8Adapter,
            "dequantize",
            staticmethod(fail_dequantize),
        )
        transform = LoRATransform(factors)
        transform.apply(param)
        torch.cuda.synchronize()

        assert param.data.qdata.data_ptr() == qdata_ptr
        assert param.data.scale.data_ptr() == scale_ptr
        assert param.data.act_quant_scale.data_ptr() == act_scale_ptr
        torch.testing.assert_close(
            param.data.dequantize().to(torch.float32),
            expected.dequantize().to(torch.float32),
            # Triton's tiled BF16 GEMM may accumulate differently from addmm.
            # Both results are then independently rounded to FP8.
            rtol=0.13,
            atol=0.05,
        )

    @CUDA
    def test_triton_lora_transform_repairs_zero_weight(self) -> None:
        f8 = _make_static_float8().cuda()
        f8.qdata.zero_()
        f8.scale.fill_(torch.finfo(torch.float32).eps)
        param = nn.Parameter(f8, requires_grad=False)
        transform = LoRATransform(
            [
                ScaledLoRAFactor.from_tensors(
                    torch.zeros(4, 16),
                    torch.zeros(16, 4),
                    1.0,
                )
            ]
        )

        transform.apply(param)
        torch.cuda.synchronize()

        dequantized = param.data.dequantize().to(torch.float32)
        assert torch.isfinite(dequantized).all()
        assert torch.count_nonzero(dequantized).item() == 0
        assert torch.count_nonzero(param.data.scale).item() == 1

    def test_merge_lora_preserves_calibrated_activation_scale(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(
                    16, 16, bias=False, dtype=torch.bfloat16
                )

        model = M()
        model.lin.weight.requires_grad = False
        model.lin.weight = nn.Parameter(
            _make_static_float8(), requires_grad=False
        )
        original_qdata = model.lin.weight.data.qdata.view(torch.uint8).clone()
        original_act_scale = model.lin.weight.data.act_quant_scale.clone()
        original_act_scale_ptr = model.lin.weight.data.act_quant_scale.data_ptr()
        lora = LoRA.from_state_dict(
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
        assert (
            model.lin.weight.data.act_quant_scale.data_ptr()
            == original_act_scale_ptr
        )
        assert torch.equal(
            model.lin.weight.data.act_quant_scale, original_act_scale
        )

    @CUDA
    def test_h2d_d2h_round_trip_preserves_wrapper_and_all_storage(self) -> None:
        prototype_cls, _, _, _ = _static_float8_modules()
        pinned_param = PinnedParam(
            nn.Parameter(
                _make_static_float8(act_scale_shape=(1, 1, 1)),
                requires_grad=False,
            )
        )
        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()
        originals = tuple(t.detach().cpu().clone() for t in gpu_state.storage)

        assert isinstance(gpu_param.data, prototype_cls)
        assert gpu_param.data.qdata.is_cuda
        assert gpu_param.data.scale.is_cuda
        assert gpu_param.data.act_quant_scale.is_cuda
        assert tuple(gpu_param.data.block_size) == pinned_param.pinned_state.meta.block_size
        assert gpu_param.data.dtype is torch.bfloat16
        for pinned in pinned_param.pinned_state.storage:
            pinned.zero_()
        pinned_param.copy_to_cpu(gpu_state, non_blocking=True)
        torch.cuda.synchronize()
        for restored, original in zip(
            pinned_param.pinned_state.storage, originals, strict=True
        ):
            assert torch.equal(
                restored.reshape(-1).view(torch.uint8),
                original.reshape(-1).view(torch.uint8),
            )

    @CUDA
    @pytest.mark.parametrize("input_shape", [(32, 64), (2, 16, 64)])
    def test_stock_linear_accepts_scalar_scale_for_2d_and_3d(
        self, input_shape: tuple[int, ...]
    ) -> None:
        layer = nn.Linear(64, 128, bias=True, dtype=torch.bfloat16)
        layer.weight.requires_grad = False
        layer.weight = nn.Parameter(
            _make_static_float8(rows=128, cols=64, act_scale_shape=()),
            requires_grad=False,
        )
        offloader = _make_model_offloader(layer)

        try:
            x = torch.randn(*input_shape, dtype=torch.bfloat16, device="cuda")
            with activated_model(offloader, "cuda") as active:
                scale_before = active.weight.data.act_quant_scale.clone()
                scale_shape = active.weight.data.act_quant_scale.shape
                y = active(x)
                torch.cuda.synchronize()
                assert active.weight.data.act_quant_scale.shape == scale_shape
                assert torch.equal(active.weight.data.act_quant_scale, scale_before)
            assert y.shape == (*input_shape[:-1], 128)
            assert y.dtype is torch.bfloat16
        finally:
            offloader.deactivate()

    @CUDA
    def test_streamed_merge_and_routed_lora_remain_supported(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        nn.Linear(64, 64, bias=False, dtype=torch.bfloat16),
                        nn.Linear(64, 64, bias=False, dtype=torch.bfloat16),
                    ]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = block(x)
                return x

        def make_model() -> M:
            model = M()
            for block in model.blocks:
                block.weight.requires_grad = False
                block.weight = nn.Parameter(
                    _make_static_float8(rows=64, cols=64),
                    requires_grad=False,
                )
            return model

        lora = LoRA.from_state_dict(
            state_dict={
                "blocks.0.lora_A.weight": torch.randn(4, 64),
                "blocks.0.lora_B.weight": torch.randn(64, 4),
            }
        )
        stream_config = StreamConfig(
            num_resident_blocks=1,
            num_prefetch_blocks=0,
        )

        for mode in ("merge", "routed"):
            model = make_model()
            calibrated_scale = model.blocks[0].weight.data.act_quant_scale.clone()
            offloader = _make_model_offloader(model, blocks_attr=["blocks"])
            try:
                x = torch.randn(2, 8, 64, dtype=torch.bfloat16, device="cuda")
                with activated_model(
                    offloader,
                    "cuda",
                    loras=[lora],
                    lora_strengths=[0.25],
                    lora_mode=mode,
                    stream_config=stream_config,
                ) as active:
                    assert torch.equal(
                        active.blocks[0].weight.data.act_quant_scale.cpu(),
                        calibrated_scale,
                    )
                    y = active(x)
                    torch.cuda.synchronize()
                assert y.shape == (2, 8, 64)
                assert y.dtype is torch.bfloat16
            finally:
                offloader.deactivate()

    @CUDA
    def test_streamed_block_compile_handles_changing_shapes(self) -> None:
        class M(nn.Module):
            def __init__(self, weights: list[torch.Tensor]) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [nn.Linear(64, 64, bias=False, dtype=torch.bfloat16) for _weight in weights]
                )
                for block, weight in zip(self.blocks, weights, strict=True):
                    block.weight.requires_grad = False
                    block.weight = nn.Parameter(
                        _make_static_float8(
                            rows=64,
                            cols=64,
                            weight=weight.clone(),
                        ),
                        requires_grad=False,
                    )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = block(x)
                return x

        torch.manual_seed(0)
        weights = [
            torch.randn(64, 64, dtype=torch.bfloat16),
            torch.randn(64, 64, dtype=torch.bfloat16),
        ]
        inputs = [
            torch.randn(2, 8, 64, dtype=torch.bfloat16),
            torch.randn(2, 12, 64, dtype=torch.bfloat16),
        ]
        stream_config = StreamConfig(
            num_resident_blocks=1,
            num_prefetch_blocks=0,
        )

        eager_model = M(weights)
        eager_offloader = _make_model_offloader(
            eager_model,
            blocks_attr=["blocks"],
        )
        try:
            with activated_model(
                eager_offloader,
                "cuda",
                stream_config=stream_config,
            ):
                with torch.inference_mode():
                    expected = [eager_model(x.cuda()).cpu() for x in inputs]
        finally:
            eager_offloader.deactivate()

        compiled_model = M(weights)
        compiled_offloader = _make_model_offloader(
            compiled_model,
            blocks_attr=["blocks"],
            block_compile=BlockCompileConfig(),
        )
        try:
            with activated_model(
                compiled_offloader,
                "cuda",
                stream_config=stream_config,
            ):
                with torch.inference_mode():
                    actual = [compiled_model(x.cuda()).cpu() for x in inputs]
        finally:
            compiled_offloader.deactivate()

        for actual_value, expected_value in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                actual_value,
                expected_value,
                rtol=0.03,
                atol=0.03,
            )
