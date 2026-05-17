"""Tests for TorchAO NVFP4 adapter integration."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torch_offload import LoRA, ModelOffloader, PinnedWeights, merge_lora
from torch_offload.nvfp4_adapter import Nvfp4Adapter
from torch_offload.pinned_buffer import PinnedParamBuffer, storage_key
from torch_offload.streamed_weights import _layout_signature

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _nvfp4_modules():
    pytest.importorskip("numpy")
    mod = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    return mod.NVFP4Tensor, mod.QuantizeTensorToNVFP4Kwargs


def _make_nvfp4(
    *,
    rows: int = 16,
    cols: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    dynamic_activation: bool = True,
    swizzled: bool = False,
) -> torch.Tensor:
    nvfp4_tensor_cls, kwargs_cls = _nvfp4_modules()
    act_quant_kwargs = (
        kwargs_cls(
            is_swizzled_scales=swizzled,
            use_dynamic_per_tensor_scale=True,
        )
        if dynamic_activation
        else None
    )
    return nvfp4_tensor_cls.to_nvfp4(
        torch.randn(rows, cols, dtype=dtype),
        per_tensor_scale=torch.tensor(0.01, dtype=torch.float32),
        is_swizzled_scales=swizzled,
        use_triton_kernel=False,
        act_quant_kwargs=act_quant_kwargs,
    )


class TestNvfp4Adapter:
    def test_matches_nvfp4_only(self) -> None:
        qt = _make_nvfp4()
        assert Nvfp4Adapter.matches(qt)
        assert not Nvfp4Adapter.matches(torch.zeros(16, 16, dtype=torch.bfloat16))

    def test_pin_preserves_storage_and_metadata(self) -> None:
        nvfp4_tensor_cls, _ = _nvfp4_modules()
        qt = _make_nvfp4(swizzled=True)
        p = nn.Parameter(qt, requires_grad=False)
        buf = PinnedParamBuffer("w", p)

        pinned = buf.cpu_param.data
        assert isinstance(pinned, nvfp4_tensor_cls)
        assert pinned.qdata.is_pinned()
        assert pinned.scale.is_pinned()
        assert pinned.per_tensor_scale is not None
        assert pinned.per_tensor_scale.is_pinned()
        assert pinned.qdata.data_ptr() == buf.pinned_state.qdata.data_ptr()
        assert pinned.scale.data_ptr() == buf.pinned_state.scale.data_ptr()
        assert pinned.per_tensor_scale.data_ptr() == buf.pinned_state.per_tensor_scale.data_ptr()
        assert pinned.block_size == qt.block_size
        assert pinned.orig_dtype == qt.orig_dtype
        assert pinned.is_swizzled_scales == qt.is_swizzled_scales
        assert pinned.use_triton_kernel == qt.use_triton_kernel
        assert pinned.act_quant_kwargs == qt.act_quant_kwargs
        assert buf.compute_dtype is torch.bfloat16

    def test_transposed_qdata_stride_is_preserved(self) -> None:
        qt = _make_nvfp4(rows=16, cols=32).t()
        buf = PinnedParamBuffer("w", nn.Parameter(qt, requires_grad=False))
        pinned = buf.cpu_param.data

        assert pinned.shape == qt.shape
        assert pinned.qdata.stride() == qt.qdata.stride()
        assert pinned.scale.stride() == qt.scale.stride()
        assert pinned.dequantize().shape == qt.dequantize().shape

    def test_storage_key_tracks_optional_scale_storage(self) -> None:
        qt = _make_nvfp4()
        key = storage_key(qt)
        assert key[0] == "torchao-nvfp4"
        assert key == storage_key(qt)

    def test_layout_signature_ignores_storage_identity(self) -> None:
        p1 = nn.Parameter(_make_nvfp4(), requires_grad=False)
        p2 = nn.Parameter(_make_nvfp4(), requires_grad=False)

        assert _layout_signature(p1) == _layout_signature(p2)

    def test_layout_signature_tracks_activation_quantization(self) -> None:
        with_activation = nn.Parameter(
            _make_nvfp4(dynamic_activation=True), requires_grad=False
        )
        weight_only = nn.Parameter(
            _make_nvfp4(dynamic_activation=False), requires_grad=False
        )

        assert _layout_signature(with_activation) != _layout_signature(weight_only)

    def test_no_cpu_round_trip_or_trainable_swap_capability(self) -> None:
        buf = PinnedParamBuffer(
            "w",
            nn.Parameter(_make_nvfp4(), requires_grad=True),
        )
        state = buf.allocate_gpu_storage(torch.device("cpu"))

        with pytest.raises(NotImplementedError, match="CPU round-trip"):
            buf.copy_to_cpu(state)
        with pytest.raises(NotImplementedError, match="Parameter.data-swap"):
            buf.validate_parameter_data_swap_target("w")

    def test_merge_lora_rejects_nvfp4_weight(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)

        model = M()
        model.lin.weight.requires_grad = False
        model.lin.weight = nn.Parameter(
            _make_nvfp4(dynamic_activation=False),
            requires_grad=False,
        )
        lora = LoRA(
            state_dict={
                "lin.lora_A.weight": torch.randn(4, 16),
                "lin.lora_B.weight": torch.randn(16, 4),
            }
        )

        with pytest.raises(ValueError, match="Nvfp4Adapter.*routed LoRA"):
            merge_lora(model, [(lora, 1.0)])

    @CUDA
    def test_load_to_gpu_preserves_wrapper(self) -> None:
        nvfp4_tensor_cls, _ = _nvfp4_modules()
        buf = PinnedParamBuffer(
            "w",
            nn.Parameter(_make_nvfp4(swizzled=True), requires_grad=False),
        )

        gpu_param = buf.load_to_gpu(torch.device("cuda"))
        torch.cuda.synchronize()

        assert isinstance(gpu_param.data, nvfp4_tensor_cls)
        assert gpu_param.data.qdata.is_cuda
        assert gpu_param.data.scale.is_cuda
        assert gpu_param.data.per_tensor_scale is not None
        assert gpu_param.data.per_tensor_scale.is_cuda
        assert gpu_param.data.block_size == buf.cpu_param.data.block_size
        assert gpu_param.data.orig_dtype == buf.cpu_param.data.orig_dtype
        assert gpu_param.data.is_swizzled_scales == buf.cpu_param.data.is_swizzled_scales
        assert torch.equal(gpu_param.data.qdata.cpu(), buf.cpu_param.data.qdata)
        assert torch.equal(gpu_param.data.scale.cpu(), buf.cpu_param.data.scale)
        assert torch.equal(
            gpu_param.data.per_tensor_scale.cpu(),
            buf.cpu_param.data.per_tensor_scale,
        )

    @CUDA
    def test_pinned_weights_cuda_forward_dynamic_nvfp4(self) -> None:
        nvfp4_mod = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
        _, kwargs_cls = _nvfp4_modules()
        layer = nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
        layer.weight.requires_grad = False
        weight = layer.weight.detach().contiguous()
        layer.weight = nn.Parameter(
            nvfp4_mod.NVFP4Tensor.to_nvfp4(
                weight,
                per_tensor_scale=nvfp4_mod.per_tensor_amax_to_scale(
                    torch.max(torch.abs(weight))
                ),
                is_swizzled_scales=True,
                use_triton_kernel=False,
                act_quant_kwargs=kwargs_cls(
                    is_swizzled_scales=True,
                    use_dynamic_per_tensor_scale=True,
                    use_triton_kernel=False,
                ),
            ),
            requires_grad=False,
        )
        strategy = PinnedWeights(layer)

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
    def test_model_offloader_routed_lora_on_dynamic_nvfp4(self) -> None:
        nvfp4_mod = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
        _, kwargs_cls = _nvfp4_modules()

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
            weight = block.weight.detach().contiguous()
            block.weight = nn.Parameter(
                nvfp4_mod.NVFP4Tensor.to_nvfp4(
                    weight,
                    per_tensor_scale=nvfp4_mod.per_tensor_amax_to_scale(
                        torch.max(torch.abs(weight))
                    ),
                    is_swizzled_scales=True,
                    use_triton_kernel=False,
                    act_quant_kwargs=kwargs_cls(
                        is_swizzled_scales=True,
                        use_dynamic_per_tensor_scale=True,
                        use_triton_kernel=False,
                    ),
                ),
                requires_grad=False,
            )
        offloader = ModelOffloader(
            model,
            layers_attr="blocks",
            blocks_to_swap=1,
            prefetch_count=0,
        )
        lora = LoRA(
            state_dict={
                "blocks.0.lora_A.weight": torch.randn(4, 128),
                "blocks.0.lora_B.weight": torch.randn(128, 4),
            }
        )
        offloader.set_loras([(lora, 0.25)], mode="routed")

        try:
            x = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
            with offloader.use("cuda") as active:
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (128, 128)
            assert y.dtype is torch.bfloat16
        finally:
            offloader.deactivate()
