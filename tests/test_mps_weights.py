"""Tests for ``torch_offload.mps_weights.MpsWeights``."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
from torch import nn

from torch_offload import ModelStrategy, MpsWeights
from torch_offload._devices import canonical_device

MPS = pytest.mark.skipif(
    not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ),
    reason="MPS required",
)


def _make_simple_model() -> nn.Module:
    model = nn.Sequential(nn.Linear(8, 16, bias=False), nn.Linear(16, 8, bias=False))
    for param in model.parameters():
        param.requires_grad = False
    return model


class _FakeMpsWeights(MpsWeights):
    """CPU-only test double for the MPS copy path.

    It lets tests exercise parameter replacement and reference lifetimes on
    machines without an MPS backend. The active tensors stay on CPU, but
    ``activate("mps")`` still takes the same copy-and-replace path.
    """

    @staticmethod
    def _check_mps_available() -> None:
        return

    @staticmethod
    def _synchronize_mps() -> None:
        return

    def _copy_tensor(
        self,
        source: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        _ = device
        return source.clone(memory_format=torch.contiguous_format)


class TestModelStrategyConformance:
    def test_isinstance_runtime_check(self) -> None:
        strategy = _FakeMpsWeights(_make_simple_model())
        try:
            assert isinstance(strategy, ModelStrategy)
        finally:
            strategy.deactivate()

    def test_cache_bytes_counts_managed_tensors(self) -> None:
        strategy = _FakeMpsWeights(_make_simple_model())
        try:
            expected = sum(param.numel() * param.element_size() for param in strategy.model.parameters())
            assert strategy.cache_bytes == expected
        finally:
            strategy.deactivate()


class TestLifecycle:
    def test_init_materializes_named_parameters(self) -> None:
        model = _make_simple_model()
        first_param = model[0].weight
        first_ref = weakref.ref(first_param)
        expected = first_param.detach().clone()
        del first_param

        strategy = _FakeMpsWeights(model)
        try:
            gc.collect()
            assert first_ref() is None
            assert torch.equal(model[0].weight, expected)
        finally:
            strategy.deactivate()

    def test_activate_without_device_raises(self) -> None:
        strategy = _FakeMpsWeights(_make_simple_model())
        try:
            with pytest.raises(ValueError, match="requires device='mps'"):
                strategy.activate()
        finally:
            strategy.deactivate()

    def test_activate_is_reentrant_noop_for_mps(self) -> None:
        strategy = _FakeMpsWeights(_make_simple_model())
        try:
            strategy.activate("mps")
            strategy.activate("mps")
        finally:
            strategy.deactivate()

    def test_rejects_unsupported_activation_device(self) -> None:
        strategy = _FakeMpsWeights(_make_simple_model())
        try:
            with pytest.raises(ValueError, match="supports MPS"):
                strategy.activate("cuda")
        finally:
            strategy.deactivate()

    def test_deactivate_before_activate_is_noop(self) -> None:
        shared = torch.randn(8, 16)
        model = nn.Module()
        model.a = nn.Parameter(shared, requires_grad=False)
        model.b = nn.Parameter(shared, requires_grad=False)
        assert model.a is not model.b

        strategy = _FakeMpsWeights(model)
        strategy.deactivate()
        assert model.a is not model.b

    def test_fake_mps_use_is_lifecycle_noop(self) -> None:
        model = _make_simple_model()
        strategy = _FakeMpsWeights(model)
        try:
            active_param = model[0].weight
            with strategy.use("mps") as active:
                assert active is model
                assert model[0].weight is active_param
            assert model[0].weight is active_param
        finally:
            strategy.deactivate()

    @MPS
    def test_real_mps_activation_leaves_model_on_mps(self) -> None:
        model = _make_simple_model()
        expected = {name: param.detach().clone() for name, param in model.named_parameters()}
        strategy = MpsWeights(model)
        try:
            assert all(param.device.type == "mps" for param in model.parameters())
            strategy.activate("mps")
            assert all(param.device.type == "mps" for param in model.parameters())
            strategy.deactivate()
            assert all(param.device.type == "mps" for param in model.parameters())
            for name, param in model.named_parameters():
                assert torch.equal(param.cpu(), expected[name])
        finally:
            strategy.deactivate()


class TestConstruction:
    def test_rejects_trainable_param(self) -> None:
        model = nn.Linear(4, 4)
        with pytest.raises(ValueError, match="cannot manage trainable parameter"):
            MpsWeights(model)

    def test_rejects_empty_model(self) -> None:
        class Empty(nn.Module):
            pass

        with pytest.raises(ValueError, match="at least one frozen parameter"):
            MpsWeights(Empty())

    def test_accepts_buffer_only_module(self) -> None:
        class BufferOnly(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("table", torch.randn(8, 4))

        model = BufferOnly()
        strategy = _FakeMpsWeights(model)
        try:
            assert strategy.cache_bytes == 8 * 4 * 4
        finally:
            strategy.deactivate()

    def test_include_buffers_false_leaves_buffer_unmanaged(self) -> None:
        class WithBuffer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4), requires_grad=False)
                self.register_buffer("table", torch.randn(8, 4))

        model = WithBuffer()
        table = model.table
        strategy = _FakeMpsWeights(model, include_buffers=False)
        try:
            assert model.table is table
        finally:
            strategy.deactivate()

def test_mps_device_is_canonicalized() -> None:
    assert canonical_device("mps:0") == torch.device("mps")
