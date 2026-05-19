"""Tests for shared slot helpers."""

from __future__ import annotations

import torch
from torch import nn

from torch_offload.slots import collect_module_slots, iter_param_slots


class _DistinctParamStorageTie(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = torch.randn(4, 4)
        self.a = nn.Module()
        self.b = nn.Module()
        self.a.weight = nn.Parameter(shared, requires_grad=False)
        self.b.weight = nn.Parameter(shared, requires_grad=False)


def test_storage_grouping_collapses_distinct_params_sharing_storage() -> None:
    model = _DistinctParamStorageTie()

    collection = collect_module_slots(model, param_group_by="storage")

    assert len(collection.param_slot_groups) == 1
    assert sorted(slot.name for slot in collection.param_slot_groups[0]) == [
        "a.weight",
        "b.weight",
    ]


def test_object_grouping_preserves_distinct_params_sharing_storage() -> None:
    model = _DistinctParamStorageTie()

    collection = collect_module_slots(model, param_group_by="object")

    assert len(collection.param_slot_groups) == 2
    assert [slots[0].name for slots in collection.param_slot_groups] == [
        "a.weight",
        "b.weight",
    ]


def test_collect_module_slots_applies_skip_before_validation() -> None:
    model = _DistinctParamStorageTie()
    skip = {
        slot.key
        for slot in iter_param_slots(model)
        if slot.name == "a.weight"
    }
    validated: list[str] = []

    collection = collect_module_slots(
        model,
        skip_slots=skip,
        validate_param=lambda slot: validated.append(slot.name),
    )

    assert validated == ["b.weight"]
    assert [slot.name for group in collection.param_slot_groups for slot in group] == [
        "b.weight",
    ]
    skipped_slots = [
        slot for slot in iter_param_slots(model) if slot.name == "a.weight"
    ]
    assert all(slot.key not in collection.slot_filter for slot in skipped_slots)


def test_buffer_grouping_uses_storage_identity() -> None:
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            shared = torch.randn(8)
            self.a = nn.Module()
            self.b = nn.Module()
            self.a.register_buffer("table", shared)
            self.b.register_buffer("table", shared)

    collection = collect_module_slots(Model())

    assert len(collection.buffer_slot_groups) == 1
    assert sorted(slot.name for slot in collection.buffer_slot_groups[0]) == [
        "a.table",
        "b.table",
    ]
