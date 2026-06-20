"""Tests for TorchAO MX (MXFP8 / MXFP4) adapter integration."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
import torch
from torch import nn

from torch_offload import (
    LoRA,
    LoRATransform,
    ModelOffloader,
    ModelOffloaderStore,
    merge_lora,
)
from torch_offload._torchao_mx import is_supported_mx_elem_dtype
from torch_offload.mx_adapter import MxAdapter
from torch_offload.pinned_param import PinnedParam
from torch_offload.streamed_component import _param_target_layout
from torch_offload.tensor_adapter_registry import select_adapter, tensor_id

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# MXFP4 packs two elements per byte under ``torch.float4_e2m1fn_x2``, which
# only exists on new-enough torch builds. Probe it so the suite degrades to
# MXFP8-only rather than failing to import.
_FP4 = getattr(torch, "float4_e2m1fn_x2", None)

ELEM_DTYPES = [
    pytest.param(torch.float8_e4m3fn, id="mxfp8"),
    pytest.param(
        _FP4,
        id="mxfp4",
        marks=pytest.mark.skipif(
            _FP4 is None, reason="torch lacks float4_e2m1fn_x2"
        ),
    ),
]


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


def _mx_tensor_cls():
    pytest.importorskip("numpy")
    mod = pytest.importorskip("torchao.prototype.mx_formats.mx_tensor")
    return mod.MXTensor


def _mx_kwargs_cls():
    mod = pytest.importorskip("torchao.prototype.mx_formats.mx_tensor")
    return mod.QuantizeTensorToMXKwargs


def _quantize_mx(
    data: torch.Tensor,
    *,
    elem_dtype: torch.dtype,
    dynamic_activation: bool = False,
) -> torch.Tensor:
    mx_cls = _mx_tensor_cls()
    # Weight-only MX cannot run a matmul on its own; a forward needs the
    # activation-quant kwargs that the dynamic-activation MX recipe carries.
    kwargs_cls = _mx_kwargs_cls()
    act_quant_kwargs = (
        kwargs_cls(elem_dtype=elem_dtype, block_size=32)
        if dynamic_activation
        else None
    )
    return mx_cls.to_mx(
        data,
        elem_dtype,
        block_size=32,
        act_quant_kwargs=act_quant_kwargs,
    )


def _make_mx(
    *,
    elem_dtype: torch.dtype,
    rows: int = 16,
    cols: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    dynamic_activation: bool = False,
) -> torch.Tensor:
    return _quantize_mx(
        torch.randn(rows, cols, dtype=dtype),
        elem_dtype=elem_dtype,
        dynamic_activation=dynamic_activation,
    )


class TestMxAdapter:
    def test_supported_elem_dtype_gate(self) -> None:
        assert is_supported_mx_elem_dtype(torch.float8_e4m3fn)
        assert is_supported_mx_elem_dtype(torch.float8_e5m2)
        if _FP4 is not None:
            assert is_supported_mx_elem_dtype(_FP4)
        # MXFP6 (string elem dtypes) and plain dtypes are out of scope.
        assert not is_supported_mx_elem_dtype("fp6_e2m3")
        assert not is_supported_mx_elem_dtype("fp6_e3m2")
        assert not is_supported_mx_elem_dtype(torch.bfloat16)
        assert not is_supported_mx_elem_dtype(None)

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_matches_mx_only(self, elem_dtype: torch.dtype) -> None:
        qt = _make_mx(elem_dtype=elem_dtype)
        assert MxAdapter.matches(qt)
        assert not MxAdapter.matches(torch.zeros(16, 64, dtype=torch.bfloat16))

    def test_rejects_mxfp6_with_clear_error(self) -> None:
        # MXFP6 is a real MXTensor but intentionally unsupported: it must
        # not dispatch to MxAdapter, and with no other adapter matching it
        # should surface the registry's "no adapter" error.
        mx_cls = _mx_tensor_cls()
        try:
            f6 = mx_cls.to_mx(
                torch.randn(16, 64, dtype=torch.bfloat16),
                "fp6_e2m3",
                block_size=32,
            )
        except Exception:  # pragma: no cover - torchao build without fp6
            pytest.skip("this torchao build cannot construct MXFP6")
        assert not MxAdapter.matches(f6)
        with pytest.raises(NotImplementedError, match="No TensorAdapter"):
            select_adapter(f6)

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_pin_preserves_storage_and_metadata(
        self, elem_dtype: torch.dtype
    ) -> None:
        mx_cls = _mx_tensor_cls()
        qt = _make_mx(elem_dtype=elem_dtype)
        pinned_param = PinnedParam(nn.Parameter(qt, requires_grad=False))

        pinned = pinned_param.make_cpu_param().data
        assert isinstance(pinned, mx_cls)
        assert pinned.qdata.is_pinned()
        assert pinned.scale.is_pinned()
        assert pinned.qdata.data_ptr() == pinned_param.pinned_state.storage[0].data_ptr()
        assert pinned.scale.data_ptr() == pinned_param.pinned_state.storage[1].data_ptr()
        assert pinned.elem_dtype == qt.elem_dtype
        assert pinned.block_size == qt.block_size
        assert pinned.orig_dtype == qt.orig_dtype
        assert pinned.is_swizzled_scales == qt.is_swizzled_scales
        assert pinned_param.compute_dtype is torch.bfloat16

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_transposed_storage_stride_is_preserved(
        self, elem_dtype: torch.dtype
    ) -> None:
        qt = _make_mx(elem_dtype=elem_dtype, rows=16, cols=64).t()
        pinned_param = PinnedParam(nn.Parameter(qt, requires_grad=False))
        pinned = pinned_param.make_cpu_param().data

        assert pinned.shape == qt.shape
        assert pinned.qdata.stride() == qt.qdata.stride()
        assert pinned.scale.stride() == qt.scale.stride()

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_tensor_id_is_stable_and_keyed(
        self, elem_dtype: torch.dtype
    ) -> None:
        qt = _make_mx(elem_dtype=elem_dtype)
        key = tensor_id(qt)
        assert key[0] == "torchao-mx"
        assert key[1][0] == qt.qdata.device
        assert key[2][0] == qt.scale.device
        assert key == tensor_id(qt)

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_target_layout_ignores_tensor_id(
        self, elem_dtype: torch.dtype
    ) -> None:
        p1 = nn.Parameter(_make_mx(elem_dtype=elem_dtype), requires_grad=False)
        p2 = nn.Parameter(_make_mx(elem_dtype=elem_dtype), requires_grad=False)

        assert _param_target_layout(p1) == _param_target_layout(p2)

    def test_target_layout_distinguishes_mxfp8_and_mxfp4(self) -> None:
        if _FP4 is None:
            pytest.skip("torch lacks float4_e2m1fn_x2")
        p8 = nn.Parameter(
            _make_mx(elem_dtype=torch.float8_e4m3fn), requires_grad=False
        )
        p4 = nn.Parameter(_make_mx(elem_dtype=_FP4), requires_grad=False)

        assert _param_target_layout(p8) != _param_target_layout(p4)

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_dynamic_activation_metadata_is_keyed(
        self, elem_dtype: torch.dtype
    ) -> None:
        # The dynamic-activation recipe carries ``act_quant_kwargs`` (a
        # dataclass) that flows through ``metadata_key`` in tensor_id /
        # layout_signature. Exercise it on CPU so the path is covered
        # without a GPU (the forward tests that use it are CUDA-gated).
        weight_only = nn.Parameter(
            _make_mx(elem_dtype=elem_dtype, dynamic_activation=False),
            requires_grad=False,
        )
        dynamic = nn.Parameter(
            _make_mx(elem_dtype=elem_dtype, dynamic_activation=True),
            requires_grad=False,
        )

        key = tensor_id(dynamic.data)
        assert key[0] == "torchao-mx"
        assert key == tensor_id(dynamic.data)
        # Activation quantization changes the matmul dispatch, so the
        # block-pool layout must distinguish it from the weight-only base.
        assert _param_target_layout(dynamic) != _param_target_layout(weight_only)

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_no_cpu_round_trip_or_trainable_swap_capability(
        self, elem_dtype: torch.dtype
    ) -> None:
        pinned_param = PinnedParam(
            nn.Parameter(_make_mx(elem_dtype=elem_dtype), requires_grad=True),
        )
        state = pinned_param.allocate_gpu_storage(torch.device("cpu"))

        with pytest.raises(NotImplementedError, match="CPU round-trip"):
            pinned_param.copy_to_cpu(state)
        with pytest.raises(NotImplementedError, match="Parameter.data-swap"):
            pinned_param.validate_parameter_data_swap_target()

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_dequantize_requantize_preserves_representation(
        self, elem_dtype: torch.dtype
    ) -> None:
        mx = _make_mx(elem_dtype=elem_dtype, rows=16, cols=64)
        dense = MxAdapter.dequantize(mx)
        assert dense.dtype is torch.float32
        torch.testing.assert_close(
            dense, mx.dequantize(mx.orig_dtype).to(torch.float32)
        )

        # MX uses deterministic FLOOR scaling onto power-of-two (E8M0)
        # block scales, so an unmodified round trip reproduces the packed
        # bytes and scales exactly.
        again = MxAdapter.requantize(dense, like=mx)
        assert again.elem_dtype == mx.elem_dtype
        assert again.block_size == mx.block_size
        assert again.orig_dtype == mx.orig_dtype
        assert again.is_swizzled_scales == mx.is_swizzled_scales
        assert again.act_quant_kwargs == mx.act_quant_kwargs
        assert torch.equal(
            again.qdata.view(torch.uint8), mx.qdata.view(torch.uint8)
        )
        assert torch.equal(
            again.scale.view(torch.uint8), mx.scale.view(torch.uint8)
        )

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_requantize_rejects_shape_mismatch(
        self, elem_dtype: torch.dtype
    ) -> None:
        mx = _make_mx(elem_dtype=elem_dtype, rows=16, cols=64)
        with pytest.raises(ValueError, match="Cannot requantize"):
            MxAdapter.requantize(torch.randn(64, 16), like=mx)

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_lora_transform_requantizes_param_in_place(
        self, elem_dtype: torch.dtype
    ) -> None:
        mx_cls = _mx_tensor_cls()
        rows, cols, rank = 16, 64, 2
        mx = _make_mx(elem_dtype=elem_dtype, rows=rows, cols=cols)
        param = nn.Parameter(mx, requires_grad=False)
        a = torch.randn(rank, cols)
        b = torch.randn(rows, rank)
        transform = LoRATransform([(a, b, 0.5)])
        original_param = param
        original_qdata_ptr = param.data.qdata.data_ptr()

        expected_dense = mx.dequantize(mx.orig_dtype).to(torch.float32)
        expected_dense.addmm_(b.to(torch.float32), a.to(torch.float32), alpha=0.5)
        expected = MxAdapter.requantize(expected_dense, like=mx)

        transform.apply(param)

        # copy_into mutates the existing wrapper's storage in place, so the
        # Parameter object and its packed-element buffer keep their identity.
        assert param is original_param
        assert param.data.qdata.data_ptr() == original_qdata_ptr
        assert isinstance(param.data, mx_cls)
        assert torch.equal(
            param.data.qdata.view(torch.uint8), expected.qdata.view(torch.uint8)
        )
        assert torch.equal(
            param.data.scale.view(torch.uint8), expected.scale.view(torch.uint8)
        )

    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_merge_lora_merges_mx_weight(
        self, elem_dtype: torch.dtype
    ) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)

        model = M()
        model.lin.weight.requires_grad = False
        model.lin.weight = nn.Parameter(
            _make_mx(elem_dtype=elem_dtype, rows=16, cols=64),
            requires_grad=False,
        )
        # copy_into mutates the weight's storage in place, so snapshot the
        # original packed bytes rather than holding a tensor ref.
        original_qdata = model.lin.weight.data.qdata.view(torch.uint8).clone()
        lora = LoRA(
            state_dict={
                "lin.lora_A.weight": torch.randn(4, 64),
                "lin.lora_B.weight": torch.randn(16, 4),
            }
        )

        merged = merge_lora(model, [(lora, 1.0)])

        assert merged == 1
        assert not torch.equal(
            model.lin.weight.data.qdata.view(torch.uint8), original_qdata
        )

    @CUDA
    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_allocate_copy_make_gpu_param_preserves_wrapper(
        self, elem_dtype: torch.dtype
    ) -> None:
        mx_cls = _mx_tensor_cls()
        pinned_param = PinnedParam(
            nn.Parameter(_make_mx(elem_dtype=elem_dtype), requires_grad=False),
        )

        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()
        pinned = pinned_param.make_cpu_param().data

        assert isinstance(gpu_param.data, mx_cls)
        assert gpu_param.data.qdata.is_cuda
        assert gpu_param.data.scale.is_cuda
        assert gpu_param.data.elem_dtype == pinned.elem_dtype
        assert gpu_param.data.block_size == pinned.block_size
        assert gpu_param.data.orig_dtype == pinned.orig_dtype
        # Compare the raw bytes: fp8 / e8m0 dtypes carry NaN encodings that
        # break value equality, so view as uint8 for a bitwise check.
        assert torch.equal(
            gpu_param.data.qdata.view(torch.uint8).cpu(),
            pinned.qdata.view(torch.uint8),
        )
        assert torch.equal(
            gpu_param.data.scale.view(torch.uint8).cpu(),
            pinned.scale.view(torch.uint8),
        )

    @CUDA
    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_model_offloader_cuda_forward_dynamic_mx(
        self, elem_dtype: torch.dtype
    ) -> None:
        layer = nn.Linear(64, 128, bias=False, dtype=torch.bfloat16)
        layer.weight.requires_grad = False
        weight = layer.weight.detach().contiguous()
        layer.weight = nn.Parameter(
            _quantize_mx(weight, elem_dtype=elem_dtype, dynamic_activation=True),
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
    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_model_offloader_routed_lora_on_dynamic_mx(
        self, elem_dtype: torch.dtype
    ) -> None:
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
                _quantize_mx(
                    weight, elem_dtype=elem_dtype, dynamic_activation=True
                ),
                requires_grad=False,
            )
        offloader = _make_model_offloader(
            model,
            blocks_attr="blocks",
            num_resident_blocks=1,
            num_prefetch_blocks=0,
        )
        lora = LoRA(
            state_dict={
                "blocks.0.lora_A.weight": torch.randn(4, 128),
                "blocks.0.lora_B.weight": torch.randn(128, 4),
            }
        )
        offloader.set_loras([lora], strengths=[0.25], mode="routed")

        try:
            x = torch.randn(128, 128, dtype=torch.bfloat16, device="cuda")
            with offloader.use("cuda") as active:
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (128, 128)
            assert y.dtype is torch.bfloat16
        finally:
            offloader.deactivate()

    @CUDA
    @pytest.mark.parametrize("elem_dtype", ELEM_DTYPES)
    def test_streamed_mx_merge_requantizes_on_activate(
        self, elem_dtype: torch.dtype
    ) -> None:
        mx_cls = _mx_tensor_cls()

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

        model = M()
        for block in model.blocks:
            block.weight.requires_grad = False
            block.weight = nn.Parameter(
                _quantize_mx(
                    block.weight.detach().contiguous(),
                    elem_dtype=elem_dtype,
                    dynamic_activation=True,
                ),
                requires_grad=False,
            )
        mx = model.blocks[0].weight.data
        rank = 4
        a = torch.randn(rank, 64)
        b = torch.randn(64, rank)
        lora = LoRA(
            state_dict={
                "blocks.0.lora_A.weight": a,
                "blocks.0.lora_B.weight": b,
            }
        )
        # Compute the reference on CUDA, matching the device the offloader
        # merges on (CPU vs CUDA rounding can flip boundary elements). The
        # offloader merges into a byte-identical GPU copy of the original
        # weight, so move that same tensor — don't re-quantize from dense.
        mx_cuda = mx.cuda()
        expected_dense = mx_cuda.dequantize(mx.orig_dtype).to(torch.float32)
        expected_dense.addmm_(
            b.cuda().to(torch.float32), a.cuda().to(torch.float32), alpha=0.5
        )
        expected = MxAdapter.requantize(expected_dense, like=mx_cuda)

        offloader = _make_model_offloader(
            model,
            blocks_attr="blocks",
            num_resident_blocks=1,
            num_prefetch_blocks=0,
        )
        offloader.set_loras([lora], strengths=[0.5], mode="merge")

        try:
            x = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
            with offloader.use("cuda") as active:
                merged = active.blocks[0].weight.data
                assert isinstance(merged, mx_cls)
                torch.testing.assert_close(
                    merged.dequantize(mx.orig_dtype).to(torch.float32),
                    expected.dequantize(mx.orig_dtype).to(torch.float32),
                )
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (8, 64)
        finally:
            offloader.deactivate()
