"""Tests for the bitsandbytes 8-bit (``Int8Params`` / LLM.int8) adapter."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from torch_offload import (
    LoRA,
    ModelOffloader,
    StreamConfig,
    merge_lora,
)
from torch_offload.bnb8bit_adapter import Bnb8bitAdapter
from torch_offload.pinned_param import PinnedParam
from torch_offload.streamed_component import _param_target_layout
from torch_offload.tensor_adapter_registry import tensor_id
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


def _make_int8(
    *,
    rows: int = 64,
    cols: int = 32,
    device: str = "cpu",
    weight: torch.Tensor | None = None,
) -> Any:
    """A quantized bitsandbytes ``Int8Params``.

    bitsandbytes 0.49+ quantizes int8 weights on CPU as well as CUDA, so
    most checks run without a GPU; only true device-transfer tests use
    :data:`CUDA`. Returns ``Any`` because ``Int8Params`` (untyped, with
    dynamically-set ``CB``/``SCB``) is opaque to the type checker.
    """
    pytest.importorskip("bitsandbytes")
    from bitsandbytes.nn import Int8Params

    int8params: Any = Int8Params
    if weight is None:
        weight = torch.randn(rows, cols, dtype=torch.float16)
    return int8params(
        weight, requires_grad=False, has_fp16_weights=False
    ).to(device)


def _unquantized_int8params(*, rows: int = 64, cols: int = 32) -> Any:
    """An ``Int8Params`` never moved onto a device, so ``CB``/``SCB`` are None.

    The shape a config-built meta skeleton's int8 weight has: module
    replacement ran but no quantization did, leaving a structural placeholder.
    """
    pytest.importorskip("bitsandbytes")
    from bitsandbytes.nn import Int8Params

    int8params: Any = Int8Params
    return int8params(
        torch.randn(rows, cols, dtype=torch.float16),
        requires_grad=False,
        has_fp16_weights=False,
    )


class TestBnb8bitAdapter:
    def test_matches_int8_only(self) -> None:
        p = _make_int8()
        assert Bnb8bitAdapter.matches(p)
        assert not Bnb8bitAdapter.matches(
            torch.zeros(64, 32, dtype=torch.float16)
        )

    def test_matches_accepts_unquantized_placeholder(self) -> None:
        # An Int8Params with CB/SCB None (an unquantized meta-skeleton
        # placeholder) is a legitimate bind target. matches is pure type
        # recognition, so it accepts it; the "must be pre-forward / quantized"
        # guard moves to the pin/read path (test_pin_rejects_post_forward).
        assert Bnb8bitAdapter.matches(_unquantized_int8params())

    def test_pin_rejects_post_forward(self) -> None:
        # After the first forward bitsandbytes nulls CB/SCB on the weight
        # (state migrates to the module). Pinning decomposes CB/SCB, so such a
        # weight still fails loudly — at the read path now, not at matches.
        p = _make_int8()
        p.CB = None
        p.SCB = None
        with pytest.raises(RuntimeError, match="before the first forward"):
            PinnedParam(p)

    def test_pin_preserves_storage_and_metadata(self) -> None:
        pytest.importorskip("bitsandbytes")
        from bitsandbytes.nn import Int8Params

        p = _make_int8()
        pinned_param = PinnedParam(p)

        # make_cpu_param() returns the Int8Params itself (a Parameter); CB is
        # the int8 weight, SCB the per-row scale.
        pinned = pinned_param.make_cpu_param()
        assert isinstance(pinned, Int8Params)
        assert pinned.CB.is_pinned()
        assert pinned.SCB.is_pinned()
        assert pinned.CB.data_ptr() == pinned_param.pinned_state.data.data_ptr()
        assert pinned.SCB.data_ptr() == pinned_param.pinned_state.scb.data_ptr()
        assert pinned.CB.dtype == torch.int8
        assert pinned_param.compute_dtype is torch.float16
        assert torch.equal(
            Bnb8bitAdapter.dequantize(pinned), Bnb8bitAdapter.dequantize(p)
        )

    def test_tensor_id_tracks_cb_and_scb(self) -> None:
        p = _make_int8()
        key = tensor_id(p)
        assert key[0] == "bnb8bit"
        assert key[1][0] == p.CB.device   # CB identity
        assert key[2][0] == p.SCB.device  # SCB identity
        assert key == tensor_id(p)
        assert key != tensor_id(_make_int8())

    def test_target_layout_ignores_tensor_id(self) -> None:
        p1 = _make_int8()
        p2 = _make_int8()

        assert _param_target_layout(p1) == _param_target_layout(p2)

    def test_bind_layout_matches_real_and_placeholder(self) -> None:
        # A config-built placeholder (CB None) must bind against a store pinned
        # from a real quantized param. int8 isn't packed, so both sides' bind
        # layout is simply the logical [out, in] shape (no quant_state branch,
        # unlike 4-bit).
        real = _make_int8(rows=64, cols=32)
        placeholder = _unquantized_int8params(rows=64, cols=32)
        assert Bnb8bitAdapter.bind_layout_signature(real) == ((64, 32),)
        assert Bnb8bitAdapter.bind_layout_signature(
            placeholder
        ) == Bnb8bitAdapter.bind_layout_signature(real)

    def test_no_cpu_round_trip_or_trainable_swap_capability(self) -> None:
        pinned_param = PinnedParam(_make_int8())
        state = pinned_param.allocate_gpu_storage(torch.device("cpu"))

        with pytest.raises(NotImplementedError, match="CPU round-trip"):
            pinned_param.copy_to_cpu(state)
        with pytest.raises(NotImplementedError, match="Parameter.data-swap"):
            pinned_param.validate_parameter_data_swap_target()

    def test_dequantize_requantize_preserves_representation(self) -> None:
        p = _make_int8()
        dense = Bnb8bitAdapter.dequantize(p)
        assert dense.dtype is torch.float16
        assert tuple(dense.shape) == tuple(p.CB.shape)

        again = Bnb8bitAdapter.requantize(dense, like=p)
        assert again.CB.dtype == torch.int8
        assert tuple(again.CB.shape) == tuple(p.CB.shape)
        assert tuple(again.SCB.shape) == tuple(p.SCB.shape)
        # Re-quantizing a weight's own dequantized (on-grid) values
        # reproduces it to within int8 + fp16 rounding.
        torch.testing.assert_close(
            Bnb8bitAdapter.dequantize(again), dense, rtol=2e-2, atol=2e-2
        )

    def test_requantize_rejects_shape_mismatch(self) -> None:
        p = _make_int8(rows=64, cols=32)
        with pytest.raises(ValueError, match="Cannot requantize"):
            Bnb8bitAdapter.requantize(torch.randn(32, 64), like=p)

    def test_requantize_large_values_do_not_overflow_to_nan(self) -> None:
        # A merged value past fp16 max (65504) must not poison the row: quant
        # happens in fp32, so the scale stays finite and dequant has no NaN.
        p = _make_int8()
        dense = Bnb8bitAdapter.dequantize(p)
        dense[0, 0] = 70000.0

        again = Bnb8bitAdapter.requantize(dense, like=p)
        assert torch.isfinite(again.SCB).all()
        assert torch.isfinite(Bnb8bitAdapter.dequantize(again)).all()

    def test_copy_into_preserves_target_identity(self) -> None:
        target = _make_int8()
        cb_ptr = target.CB.data_ptr()
        scb_ptr = target.SCB.data_ptr()

        src = Bnb8bitAdapter.requantize(
            Bnb8bitAdapter.dequantize(target) + 0.2, like=target
        )
        Bnb8bitAdapter.copy_into(src, target=target)

        assert target.CB.data_ptr() == cb_ptr
        assert target.SCB.data_ptr() == scb_ptr
        assert torch.equal(target.CB, src.CB)
        assert torch.equal(target.SCB, src.SCB)

    def test_merge_lora_merges_int8_weight(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(32, 64, bias=False, dtype=torch.float16)

        model = M()
        model.lin.weight.requires_grad = False
        # Int8Params is already a Parameter — assign it directly.
        model.lin.weight = _make_int8(rows=64, cols=32)
        original_cb = model.lin.weight.CB.clone()
        lora = LoRA.from_state_dict(
            state_dict={
                "lin.lora_A.weight": torch.randn(4, 32),
                "lin.lora_B.weight": torch.randn(64, 4),
            }
        )

        merged = merge_lora(model, [(lora, 1.0)])

        assert merged == 1
        assert not torch.equal(model.lin.weight.CB, original_cb)

    @CUDA
    def test_merge_lora_int8_on_cuda(self) -> None:
        # Production LoRA merge runs the requantize on the GPU int8 kernel,
        # which requires fp16 input — a CPU-only merge test misses that.
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(32, 64, bias=False, dtype=torch.float16)

        model = M().to("cuda")
        model.lin.weight = _make_int8(rows=64, cols=32, device="cuda")
        original_cb = model.lin.weight.CB.clone()
        lora = LoRA.from_state_dict(
            state_dict={
                "lin.lora_A.weight": torch.randn(4, 32),
                "lin.lora_B.weight": torch.randn(64, 4),
            }
        )

        merged = merge_lora(model, [(lora, 1.0)])

        assert merged == 1
        assert torch.isfinite(model.lin.weight.SCB).all()
        assert not torch.equal(model.lin.weight.CB, original_cb)

    def test_tied_int8_weights_rejected(self) -> None:
        # int8 quant state migrates onto the module on first forward, so a
        # single shared wrapper cannot serve two tied modules; reject at pin.
        shared = _make_int8(rows=64, cols=64)

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Linear(64, 64, bias=False)
                self.b = nn.Linear(64, 64, bias=False)

        model = M()
        model.a.weight = shared
        model.b.weight = shared  # tied

        with pytest.raises(NotImplementedError, match="Tied"):
            ModelOffloader.from_module(model)

    @CUDA
    def test_allocate_copy_make_gpu_param_preserves_wrapper(self) -> None:
        pytest.importorskip("bitsandbytes")
        from bitsandbytes.nn import Int8Params

        pinned_param = PinnedParam(_make_int8(device="cuda"))

        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()
        pinned = pinned_param.make_cpu_param()

        assert isinstance(gpu_param, Int8Params)
        assert gpu_param.CB.is_cuda
        assert gpu_param.SCB.is_cuda
        assert gpu_param.CB.dtype == torch.int8
        assert torch.equal(gpu_param.CB.cpu(), pinned.CB)
        assert torch.equal(gpu_param.SCB.cpu(), pinned.SCB)

    @CUDA
    def test_model_offloader_cuda_forward_int8_multi_cycle(self) -> None:
        bnb = pytest.importorskip("bitsandbytes")
        layer = bnb.nn.Linear8bitLt(
            64, 128, bias=False, has_fp16_weights=False
        ).to("cuda")

        # Independent reference from a twin built from the SAME pre-forward
        # CB/SCB, so we never forward `layer` before the store pins it.
        cb = layer.weight.CB.clone()
        scb = layer.weight.SCB.clone()
        ref_layer = bnb.nn.Linear8bitLt(
            64, 128, bias=False, has_fp16_weights=False
        ).to("cuda")
        ref_layer._parameters["weight"] = bnb.nn.Int8Params(
            cb.clone(), requires_grad=False, has_fp16_weights=False,
            CB=cb.clone(), SCB=scb.clone(),
        )
        x = torch.randn(8, 64, dtype=torch.float16, device="cuda")
        reference = ref_layer(x)

        # `layer` is still pre-forward here, so the store pins CB/SCB.
        strategy = _make_model_offloader(layer)
        try:
            # Multiple activation cycles exercise the init_8bit_state re-fire:
            # each activation installs a fresh CB-bearing weight, so the
            # module state is repopulated without the adapter touching it.
            for _ in range(3):
                with activated_model(strategy, "cuda") as active:
                    y = active(x)
                    torch.cuda.synchronize()
                assert y.shape == (8, 128)
                assert y.dtype is torch.float16
                torch.testing.assert_close(y, reference)
        finally:
            strategy.deactivate()

    @CUDA
    def test_model_offloader_streamed_int8_blocks(self) -> None:
        # Streamed/pooled block rotation: blocks share a GPU pool and are
        # reconstructed on rotation. Exercises the int8 init re-fire under
        # the pooled lifecycle (distinct from the whole-model path above).
        bnb = pytest.importorskip("bitsandbytes")

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(
                    [
                        bnb.nn.Linear8bitLt(
                            128, 128, bias=False, has_fp16_weights=False
                        ),
                        bnb.nn.Linear8bitLt(
                            128, 128, bias=False, has_fp16_weights=False
                        ),
                    ]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    x = block(x)
                return x

        model = M().to("cuda")
        # Reference twin from each block's pre-forward CB/SCB.
        caps = [(b.weight.CB.clone(), b.weight.SCB.clone()) for b in model.blocks]
        ref_model = M().to("cuda")
        for block, (cb, scb) in zip(ref_model.blocks, caps, strict=True):
            block._parameters["weight"] = bnb.nn.Int8Params(
                cb.clone(), requires_grad=False, has_fp16_weights=False,
                CB=cb.clone(), SCB=scb.clone(),
            )
        x = torch.randn(128, 128, dtype=torch.float16, device="cuda")
        reference = ref_model(x)

        offloader = _make_model_offloader(
            model,
            blocks_attr=["blocks"],
        )
        try:
            for _ in range(3):
                with activated_model(offloader,
                    "cuda", stream_config=StreamConfig(num_prefetch_blocks=0)
                ) as active:
                    y = active(x)
                    torch.cuda.synchronize()
                torch.testing.assert_close(y, reference)
        finally:
            offloader.deactivate()
