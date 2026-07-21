"""Tests for TorchAO tile-packed int4 (``Int4TilePackedTo4dTensor``).

Creating a tile-packed int4 weight runs TorchAO's CUDA-only int4-pack
kernel, so the whole module requires CUDA. Re-wrapping the already-packed
bytes (the pinned-CPU path the adapter exercises) works on CPU, but the
source tensor still has to be built on the GPU.
"""

from __future__ import annotations


import pytest
import torch
from torch import nn

from torch_offload import (
    LoRA,
    ModelOffloader,
    StreamConfig,
    merge_lora,
)
from torch_offload.int4_tile_adapter import Int4TilePackedAdapter
from torch_offload.pinned_param import PinnedParam
from torch_offload.streamed_component import _param_target_layout
from torch_offload.tensor_adapter_registry import select_adapter, tensor_id
from tests.conftest import activated_model

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required (tile-packed int4 uses a CUDA-only pack kernel)",
)


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


def _int4_tile_config() -> object:
    pytest.importorskip("torchao")
    try:
        from torchao.quantization import Int4WeightOnlyConfig
        from torchao.quantization.quantize_.workflows import Int4PackingFormat
    except ImportError as exc:
        pytest.skip(f"torchao int4 API unavailable: {exc}")
    return Int4WeightOnlyConfig(int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D)


def _int4_tile_cls() -> type:
    pytest.importorskip("torchao")
    try:
        from torchao.quantization.quantize_.workflows import (
            Int4TilePackedTo4dTensor,
        )
    except ImportError as exc:
        pytest.skip(f"torchao Int4TilePackedTo4dTensor unavailable: {exc}")
    return Int4TilePackedTo4dTensor


def _make_int4_tile(
    *,
    rows: int = 256,
    cols: int = 256,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    from torchao.quantization import quantize_

    cfg = _int4_tile_config()
    layer = nn.Linear(cols, rows, bias=False).to(dtype).cuda()
    quantize_(layer, cfg)
    return layer.weight.data


class TestInt4TilePackedAdapter:
    def test_matches_and_dispatches_int4_tile_only(self) -> None:
        qt = _make_int4_tile()
        assert Int4TilePackedAdapter.matches(qt)
        assert not Int4TilePackedAdapter.matches(torch.zeros(16, 16, dtype=torch.bfloat16))
        assert isinstance(select_adapter(qt), Int4TilePackedAdapter)

    def test_pin_preserves_storage_and_metadata(self) -> None:
        int4_cls = _int4_tile_cls()
        qt = _make_int4_tile()
        pinned_param = PinnedParam(nn.Parameter(qt, requires_grad=False))

        pinned = pinned_param.make_cpu_param().data
        assert isinstance(pinned, int4_cls)
        assert pinned.qdata.is_pinned()
        assert pinned.scale_and_zero.is_pinned()
        assert pinned.qdata.data_ptr() == pinned_param.pinned_state.storage[0].data_ptr()
        assert pinned.scale_and_zero.data_ptr() == pinned_param.pinned_state.storage[1].data_ptr()
        # Logical shape is preserved even though qdata is packed to a
        # different (4-D) shape.
        assert tuple(pinned.shape) == tuple(qt.shape)
        assert tuple(pinned.qdata.shape) == tuple(qt.qdata.shape)
        assert pinned.block_size == qt.block_size
        assert pinned.dtype == qt.dtype
        assert pinned_param.compute_dtype is torch.bfloat16

    def test_tensor_id_tracks_buffers(self) -> None:
        qt = _make_int4_tile()
        key = tensor_id(qt)
        assert key[0] == "torchao-int4-tile-packed"
        assert key[1][0] == qt.qdata.device
        assert key[2][0] == qt.scale_and_zero.device
        assert key == tensor_id(qt)
        assert key != tensor_id(_make_int4_tile())

    def test_target_layout_ignores_tensor_id(self) -> None:
        p1 = nn.Parameter(_make_int4_tile(), requires_grad=False)
        p2 = nn.Parameter(_make_int4_tile(), requires_grad=False)

        assert _param_target_layout(p1) == _param_target_layout(p2)

    def test_no_cpu_round_trip_or_trainable_swap_capability(self) -> None:
        pinned_param = PinnedParam(
            nn.Parameter(_make_int4_tile(), requires_grad=True),
        )
        state = pinned_param.allocate_gpu_storage(torch.device("cpu"))

        with pytest.raises(NotImplementedError, match="CPU round-trip"):
            pinned_param.copy_to_cpu(state)
        with pytest.raises(NotImplementedError, match="Parameter.data-swap"):
            pinned_param.validate_parameter_data_swap_target()

    def test_merge_lora_rejects_int4_tile_weight(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(256, 256, bias=False, dtype=torch.bfloat16)

        model = M()
        model.lin.weight.requires_grad = False
        model.lin.weight = nn.Parameter(_make_int4_tile(), requires_grad=False)
        lora = LoRA.from_state_dict(
            state_dict={
                "lin.lora_A.weight": torch.randn(8, 256),
                "lin.lora_B.weight": torch.randn(256, 8),
            }
        )

        with pytest.raises(ValueError, match="Int4TilePackedAdapter.*routed LoRA"):
            merge_lora(model, [(lora, 1.0)])

    def test_allocate_copy_make_gpu_param_preserves_wrapper(self) -> None:
        int4_cls = _int4_tile_cls()
        qt = _make_int4_tile()
        pinned_param = PinnedParam(nn.Parameter(qt, requires_grad=False))

        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()
        pinned = pinned_param.make_cpu_param().data

        assert isinstance(gpu_param.data, int4_cls)
        assert gpu_param.data.qdata.is_cuda
        assert gpu_param.data.scale_and_zero.is_cuda
        assert tuple(gpu_param.data.shape) == tuple(pinned.shape)
        assert gpu_param.data.block_size == pinned.block_size
        assert torch.equal(gpu_param.data.qdata.cpu(), pinned.qdata)
        assert torch.equal(gpu_param.data.scale_and_zero.cpu(), pinned.scale_and_zero)

    def test_model_offloader_cuda_forward(self) -> None:
        layer = nn.Linear(256, 256, bias=False, dtype=torch.bfloat16)
        layer.weight.requires_grad = False
        layer.weight = nn.Parameter(_make_int4_tile(), requires_grad=False)
        strategy = _make_model_offloader(layer)

        try:
            x = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
            with activated_model(strategy, "cuda") as active:
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (64, 256)
            assert y.dtype is torch.bfloat16
        finally:
            strategy.deactivate()

    def test_model_offloader_cuda_forward_matches_reference(self) -> None:
        # The offloaded weight must produce the same output as the original
        # GPU-resident tile-packed weight (reconstruction is byte-exact).
        layer = nn.Linear(256, 256, bias=False, dtype=torch.bfloat16)
        layer.weight.requires_grad = False
        qt = _make_int4_tile()
        layer.weight = nn.Parameter(qt, requires_grad=False)
        x = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
        ref = torch.nn.functional.linear(x, qt)

        strategy = _make_model_offloader(layer)
        try:
            with activated_model(strategy, "cuda") as active:
                y = active(x)
                torch.cuda.synchronize()
            torch.testing.assert_close(y, ref)
        finally:
            strategy.deactivate()

    def test_model_offloader_routed_lora_on_int4_tile(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        nn.Linear(256, 256, bias=False, dtype=torch.bfloat16),
                        nn.Linear(256, 256, bias=False, dtype=torch.bfloat16),
                    ]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = block(x)
                return x

        model = M()
        for block in model.blocks:
            block.weight.requires_grad = False
            block.weight = nn.Parameter(_make_int4_tile(), requires_grad=False)
        offloader = _make_model_offloader(
            model,
            blocks_attr=["blocks"],
        )
        lora = LoRA.from_state_dict(
            state_dict={
                "blocks.0.lora_A.weight": torch.randn(8, 256),
                "blocks.0.lora_B.weight": torch.randn(256, 8),
            }
        )
        try:
            x = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
            with activated_model(offloader,
                "cuda",
                loras=[lora],
                lora_strengths=[0.25],
                lora_mode="routed",
                stream_config=StreamConfig(num_resident_blocks=1, num_prefetch_blocks=0),
            ) as active:
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (64, 256)
            assert y.dtype is torch.bfloat16
        finally:
            offloader.deactivate()
