"""Tests for GGUFWeight tensor subclass and GgufAdapter."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torch_offload.gguf_adapter import GGUFWeight, GgufAdapter

QUANT_TYPE = 2  # arbitrary, just needs to round-trip


@pytest.fixture()
def w() -> GGUFWeight:
    return GGUFWeight(torch.zeros(64, dtype=torch.uint8), quant_type=QUANT_TYPE)


class TestQuantTypeSurvival:
    """quant_type must survive the operations in the actual pipeline path:
    creation → nn.Parameter wrapping → .data access → clone_pin."""

    def test_data_attr(self, w: GGUFWeight) -> None:
        assert isinstance(w.data, GGUFWeight)
        assert w.data.quant_type == QUANT_TYPE

    def test_nn_parameter_wrapping(self, w: GGUFWeight) -> None:
        p = nn.Parameter(w, requires_grad=False)
        assert isinstance(p.data, GGUFWeight)
        assert p.data.quant_type == QUANT_TYPE

    def test_clone(self, w: GGUFWeight) -> None:
        assert w.clone().quant_type == QUANT_TYPE

    def test_detach(self, w: GGUFWeight) -> None:
        assert w.detach().quant_type == QUANT_TYPE

    def test_contiguous(self, w: GGUFWeight) -> None:
        assert w.contiguous().quant_type == QUANT_TYPE

    def test_view(self, w: GGUFWeight) -> None:
        assert w.view(8, 8).quant_type == QUANT_TYPE

    def test_reshape(self, w: GGUFWeight) -> None:
        assert w.reshape(8, 8).quant_type == QUANT_TYPE

    def test_to_same_dtype(self, w: GGUFWeight) -> None:
        assert w.to(torch.uint8).quant_type == QUANT_TYPE

    def test_to_same_device(self, w: GGUFWeight) -> None:
        assert w.to("cpu").quant_type == QUANT_TYPE

    def test_split(self, w: GGUFWeight) -> None:
        for p in w.split(32):
            assert isinstance(p, GGUFWeight)
            assert p.quant_type == QUANT_TYPE

    def test_chunk(self, w: GGUFWeight) -> None:
        for p in w.chunk(4):
            assert isinstance(p, GGUFWeight)
            assert p.quant_type == QUANT_TYPE


class TestAdapter:
    def test_matches_gguf(self, w: GGUFWeight) -> None:
        assert GgufAdapter.matches(w)

    def test_no_match_plain_uint8(self) -> None:
        assert not GgufAdapter.matches(torch.zeros(4, dtype=torch.uint8))

    def test_no_match_dtype_changed(self, w: GGUFWeight) -> None:
        """A GGUFWeight that got .float()'d is no longer valid packed data."""
        assert not GgufAdapter.matches(w.float())

    def test_storage_key_deterministic(self, w: GGUFWeight) -> None:
        assert GgufAdapter.storage_key(w) == GgufAdapter.storage_key(w)
