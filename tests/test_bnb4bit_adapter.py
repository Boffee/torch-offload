"""Tests for bitsandbytes 4-bit (``Params4bit``) adapter integration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest
import torch
from torch import nn

from torch_offload import LoRA, ModelOffloader, ModelOffloaderStore, merge_lora
from torch_offload.bnb4bit_adapter import Bnb4bitAdapter
from torch_offload.pinned_param import PinnedParam
from torch_offload.streamed_component import _param_target_layout
from torch_offload.tensor_adapter_registry import tensor_id

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


def _make_nf4(
    *,
    rows: int = 64,
    cols: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    quant_type: str = "nf4",
    double_quant: bool = False,
    device: str = "cpu",
    weight: torch.Tensor | None = None,
) -> Any:
    """A quantized bitsandbytes ``Params4bit``.

    bitsandbytes 0.49+ quantizes 4-bit weights on CPU as well as CUDA, so
    most checks run without a GPU; only true device-transfer tests use
    :data:`CUDA`. Returns ``Any`` because ``Params4bit`` (untyped, with
    dynamically-set ``quant_state``) is opaque to the type checker.
    """
    pytest.importorskip("bitsandbytes")
    from bitsandbytes.nn import Params4bit

    params4bit: Any = Params4bit
    if weight is None:
        weight = torch.randn(rows, cols, dtype=dtype)
    return params4bit(
        weight,
        requires_grad=False,
        quant_type=quant_type,
        compress_statistics=double_quant,
        quant_storage=torch.uint8,
    ).to(device)


class TestBnb4bitAdapter:
    def test_matches_nf4_only(self) -> None:
        p = _make_nf4()
        assert Bnb4bitAdapter.matches(p)
        assert not Bnb4bitAdapter.matches(
            torch.zeros(64, 32, dtype=torch.bfloat16)
        )

    def test_matches_fp4(self) -> None:
        # One adapter covers the whole 4-bit family, not just NF4.
        assert Bnb4bitAdapter.matches(_make_nf4(quant_type="fp4"))

    def test_matches_rejects_unquantized_params4bit(self) -> None:
        pytest.importorskip("bitsandbytes")
        from bitsandbytes.nn import Params4bit

        # A Params4bit that was never moved onto a device carries no
        # quant_state — it cannot be offloaded, and matches says so loudly.
        params4bit: Any = Params4bit
        placeholder = params4bit(
            torch.randn(64, 32, dtype=torch.bfloat16), requires_grad=False
        )
        with pytest.raises(RuntimeError, match="not quantized"):
            Bnb4bitAdapter.matches(placeholder)

    def test_pin_preserves_storage_and_metadata(self) -> None:
        pytest.importorskip("bitsandbytes")
        from bitsandbytes.nn import Params4bit

        # Params4bit is itself an nn.Parameter; pass it directly (wrapping it
        # in nn.Parameter raises, since its detach() returns a plain Tensor).
        p = _make_nf4()
        pinned_param = PinnedParam(p)

        # make_cpu_param() returns the Params4bit itself (a Parameter); its
        # .data is the packed uint8 weight, quant_state holds the scales.
        pinned = pinned_param.make_cpu_param()
        assert isinstance(pinned, Params4bit)
        assert pinned.data.is_pinned()
        assert pinned.quant_state.absmax.is_pinned()
        assert pinned.data.data_ptr() == pinned_param.pinned_state.data.data_ptr()
        assert (
            pinned.quant_state.absmax.data_ptr()
            == pinned_param.pinned_state.buffers["absmax"].data_ptr()
        )
        assert pinned.quant_state.blocksize == p.quant_state.blocksize
        assert pinned.quant_state.quant_type == p.quant_state.quant_type
        assert tuple(pinned.quant_state.shape) == tuple(p.quant_state.shape)
        assert pinned_param.compute_dtype is torch.bfloat16
        assert torch.equal(
            Bnb4bitAdapter.dequantize(pinned), Bnb4bitAdapter.dequantize(p)
        )

    def test_tensor_id_tracks_packed_and_scales(self) -> None:
        p = _make_nf4()
        key = tensor_id(p)
        assert key[0] == "bnb4bit"
        assert key[1][0] == p.data.device  # packed weight identity
        assert key[2][0] == p.quant_state.absmax.device  # absmax identity
        assert key == tensor_id(p)
        assert key != tensor_id(_make_nf4())

    def test_target_layout_ignores_tensor_id(self) -> None:
        p1 = _make_nf4()
        p2 = _make_nf4()

        assert _param_target_layout(p1) == _param_target_layout(p2)

    def test_target_layout_tracks_quant_type(self) -> None:
        nf4 = _make_nf4(quant_type="nf4")
        fp4 = _make_nf4(quant_type="fp4")

        assert _param_target_layout(nf4) != _param_target_layout(fp4)

    def test_target_layout_tracks_double_quant(self) -> None:
        single = _make_nf4(double_quant=False)
        nested = _make_nf4(double_quant=True)

        assert _param_target_layout(single) != _param_target_layout(nested)

    def test_no_cpu_round_trip_or_trainable_swap_capability(self) -> None:
        pinned_param = PinnedParam(_make_nf4())
        state = pinned_param.allocate_gpu_storage(torch.device("cpu"))

        with pytest.raises(NotImplementedError, match="CPU round-trip"):
            pinned_param.copy_to_cpu(state)
        with pytest.raises(NotImplementedError, match="Parameter.data-swap"):
            pinned_param.validate_parameter_data_swap_target()

    @pytest.mark.parametrize("double_quant", [False, True])
    def test_dequantize_requantize_preserves_representation(
        self, double_quant: bool
    ) -> None:
        p = _make_nf4(double_quant=double_quant)
        dense = Bnb4bitAdapter.dequantize(p)
        assert dense.dtype is torch.float32
        assert tuple(dense.shape) == tuple(p.quant_state.shape)

        again = Bnb4bitAdapter.requantize(dense, like=p)
        assert again.quant_state.quant_type == p.quant_state.quant_type
        assert again.quant_state.blocksize == p.quant_state.blocksize
        assert again.quant_state.nested == p.quant_state.nested
        assert tuple(again.data.shape) == tuple(p.data.shape)
        if not double_quant:
            # Single-level absmax: requantizing a weight's own dequantized
            # values reproduces it exactly — the per-block max maps to the
            # codebook extreme, so absmax is recovered. With double-quant the
            # absmax is itself quantized, so this is not bit-exact (the nested
            # full round-trip is covered by the GPU-lifecycle test).
            assert torch.equal(again.data, p.data)
            assert torch.equal(again.quant_state.absmax, p.quant_state.absmax)

    def test_requantize_rejects_shape_mismatch(self) -> None:
        p = _make_nf4(rows=64, cols=32)
        with pytest.raises(ValueError, match="Cannot requantize"):
            Bnb4bitAdapter.requantize(torch.randn(32, 64), like=p)

    def test_copy_into_preserves_target_identity(self) -> None:
        target = _make_nf4()
        target_data_ptr = target.data.data_ptr()
        target_absmax_ptr = target.quant_state.absmax.data_ptr()

        src = Bnb4bitAdapter.requantize(
            Bnb4bitAdapter.dequantize(target) + 0.1, like=target
        )
        Bnb4bitAdapter.copy_into(src, target=target)

        # Storage identity preserved; contents now match the source.
        assert target.data.data_ptr() == target_data_ptr
        assert target.quant_state.absmax.data_ptr() == target_absmax_ptr
        assert torch.equal(target.data, src.data)
        assert torch.equal(target.quant_state.absmax, src.quant_state.absmax)

    def test_merge_lora_merges_bnb4bit_weight(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(32, 64, bias=False, dtype=torch.bfloat16)

        model = M()
        model.lin.weight.requires_grad = False
        # Params4bit is already a Parameter — assign it directly.
        model.lin.weight = _make_nf4(rows=64, cols=32)
        # copy_into mutates the weight's packed storage in place, so snapshot
        # the original bytes rather than holding a tensor reference.
        original_packed = model.lin.weight.data.data.clone()
        lora = LoRA(
            state_dict={
                "lin.lora_A.weight": torch.randn(4, 32),
                "lin.lora_B.weight": torch.randn(64, 4),
            }
        )

        merged = merge_lora(model, [(lora, 1.0)])

        assert merged == 1
        assert not torch.equal(model.lin.weight.data.data, original_packed)

    @CUDA
    def test_allocate_copy_make_gpu_param_preserves_wrapper(self) -> None:
        pytest.importorskip("bitsandbytes")
        from bitsandbytes.nn import Params4bit

        pinned_param = PinnedParam(_make_nf4(device="cuda"))

        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()
        # Both wrappers are Params4bit (Parameters); .data is the packed weight.
        pinned = pinned_param.make_cpu_param()

        assert isinstance(gpu_param, Params4bit)
        assert gpu_param.data.is_cuda
        assert gpu_param.quant_state.absmax.is_cuda
        assert gpu_param.quant_state.blocksize == pinned.quant_state.blocksize
        assert gpu_param.quant_state.quant_type == pinned.quant_state.quant_type
        assert torch.equal(gpu_param.data.cpu(), pinned.data)
        assert torch.equal(
            gpu_param.quant_state.absmax.cpu(), pinned.quant_state.absmax
        )

    @CUDA
    def test_double_quant_full_gpu_lifecycle_round_trip(self) -> None:
        # Double-quant (nested) carries extra nested_absmax / nested_quant_map
        # buffers plus an offset; verify they all DMA and reconstruct. Common
        # in real NF4 checkpoints (Flux, many LLMs), distinct from the flat
        # layout this model uses.
        p = _make_nf4(rows=128, cols=64, double_quant=True, device="cuda")
        assert p.quant_state.nested
        ref = Bnb4bitAdapter.dequantize(p).clone()

        pinned_param = PinnedParam(p)
        assert "nested_absmax" in pinned_param.pinned_state.buffers
        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        torch.cuda.synchronize()

        assert gpu_param.quant_state.nested
        assert torch.equal(Bnb4bitAdapter.dequantize(gpu_param), ref)

    @CUDA
    def test_model_offloader_cuda_forward_nf4(self) -> None:
        bnb = pytest.importorskip("bitsandbytes")
        layer = bnb.nn.Linear4bit(
            64,
            128,
            bias=False,
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
            quant_storage=torch.uint8,
        ).to("cuda")
        x = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
        reference = layer(x)

        strategy = _make_model_offloader(layer)
        try:
            with strategy.use("cuda") as active:
                y = active(x)
                torch.cuda.synchronize()
            assert y.shape == (8, 128)
            assert y.dtype is torch.bfloat16
            torch.testing.assert_close(y, reference)
        finally:
            strategy.deactivate()

    @CUDA
    def test_model_offloader_routed_lora_on_nf4(self) -> None:
        bnb = pytest.importorskip("bitsandbytes")

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        bnb.nn.Linear4bit(
                            128, 128, bias=False, compute_dtype=torch.bfloat16,
                            quant_type="nf4", quant_storage=torch.uint8,
                        ),
                        bnb.nn.Linear4bit(
                            128, 128, bias=False, compute_dtype=torch.bfloat16,
                            quant_type="nf4", quant_storage=torch.uint8,
                        ),
                    ]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = block(x)
                return x

        model = M().to("cuda")
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
