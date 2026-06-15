"""Tests for TorchAO Int8 (``Int8Tensor``) adapter integration."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
import torch
from torch import nn

from torch_offload import LoRA, ModelOffloader, ModelOffloaderStore, merge_lora
from torch_offload.int8_adapter import Int8Adapter
from torch_offload.pinned_param import PinnedParam
from torch_offload.streamed_component import _param_target_layout
from torch_offload.tensor_adapter_registry import select_adapter, tensor_id

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


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

    return (
        Int8DynamicActivationInt8WeightConfig(version=2)
        if dynamic_activation
        else Int8WeightOnlyConfig(version=2)
    )


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
            assert (
                pinned.zero_point.data_ptr()
                == pinned_param.pinned_state.storage[2].data_ptr()
            )
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
        with_activation = nn.Parameter(
            _make_int8(dynamic_activation=True), requires_grad=False
        )
        weight_only = nn.Parameter(
            _make_int8(dynamic_activation=False), requires_grad=False
        )

        assert _param_target_layout(with_activation) != _param_target_layout(
            weight_only
        )

    def test_no_cpu_round_trip_or_trainable_swap_capability(self) -> None:
        pinned_param = PinnedParam(
            nn.Parameter(_make_int8(), requires_grad=True),
        )
        state = pinned_param.allocate_gpu_storage(torch.device("cpu"))

        with pytest.raises(NotImplementedError, match="CPU round-trip"):
            pinned_param.copy_to_cpu(state)
        with pytest.raises(NotImplementedError, match="Parameter.data-swap"):
            pinned_param.validate_parameter_data_swap_target()

    def test_merge_lora_rejects_int8_weight(self) -> None:
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
        lora = LoRA(
            state_dict={
                "lin.lora_A.weight": torch.randn(4, 16),
                "lin.lora_B.weight": torch.randn(16, 4),
            }
        )

        with pytest.raises(ValueError, match="Int8Adapter.*routed LoRA"):
            merge_lora(model, [(lora, 1.0)])

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
            assert torch.equal(
                gpu_param.data.zero_point.cpu(), pinned.zero_point
            )

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
            with strategy.use("cuda") as active:
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
