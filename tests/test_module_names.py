"""Tests for shared name and storage-identity helpers."""

from __future__ import annotations

import torch
from torch import nn

from torch_offload.module_names import (
    named_buffer_entries,
    named_parameter_entries,
)
from torch_offload.tensor_adapter_registry import (
    buffer_storage_key,
    param_storage_key,
)


def test_named_parameter_entries_return_duplicate_module_paths() -> None:
    model = nn.Module()
    shared = nn.Linear(4, 4, bias=False)
    model.left = shared
    model.right = shared

    entries = list(named_parameter_entries(model))

    assert [(name, parent, leaf) for name, parent, leaf, _param in entries] == [
        ("left.weight", shared, "weight"),
        ("right.weight", shared, "weight"),
    ]


def test_named_buffer_entries_preserve_persistence() -> None:
    model = nn.Module()
    model.register_buffer("persistent", torch.randn(2), persistent=True)
    model.register_buffer("temporary", torch.randn(2), persistent=False)

    entries = {
        name: persistent
        for name, _parent, _leaf, _buffer, persistent in named_buffer_entries(model)
    }

    assert entries["persistent"] is True
    assert entries["temporary"] is False


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
