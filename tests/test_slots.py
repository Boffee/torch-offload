"""Tests for shared slot helpers."""

from __future__ import annotations

import torch
from torch import nn

from torch_offload.slots import (
    buffer_storage_key,
    iter_buffer_slots,
    iter_param_slots,
    param_storage_key,
)


def test_iter_param_slots_returns_duplicate_module_paths() -> None:
    model = nn.Module()
    shared = nn.Linear(4, 4, bias=False)
    model.left = shared
    model.right = shared

    slots = list(iter_param_slots(model))

    assert [slot.name for slot in slots] == ["left.weight", "right.weight"]
    assert slots[0].parent is shared
    assert slots[1].parent is shared
    assert slots[0].leaf == "weight"
    assert slots[1].leaf == "weight"
    assert slots[0].key == slots[1].key


def test_iter_buffer_slots_preserves_persistence() -> None:
    model = nn.Module()
    model.register_buffer("persistent", torch.randn(2), persistent=True)
    model.register_buffer("temporary", torch.randn(2), persistent=False)

    slots = {slot.name: slot for slot in iter_buffer_slots(model)}

    assert slots["persistent"].persistent is True
    assert slots["temporary"].persistent is False


def test_param_storage_key_groups_distinct_params_sharing_storage() -> None:
    shared = torch.randn(4, 4)
    left = nn.Parameter(shared, requires_grad=False)
    right = nn.Parameter(shared, requires_grad=False)

    assert left is not right
    assert param_storage_key(left) == param_storage_key(right)


def test_buffer_storage_key_groups_shared_storage() -> None:
    shared = torch.randn(8)
    alias = shared.view_as(shared)

    assert shared is not alias
    assert buffer_storage_key(shared) == buffer_storage_key(alias)
