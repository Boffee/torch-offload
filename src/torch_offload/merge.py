"""Permanent LoRA merge into model weights.

Merges LoRA deltas directly into model parameters in-place, supporting
both unquantized (bf16/fp16/fp32) and quanto-quantized weights. For
quantized weights, the merge dequantizes, applies the delta, and
requantizes — this is lossy but standard practice.

Unlike :class:`LoRATransform` (which merges at DMA time and is
reversible), this is a one-shot permanent modification.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from ._quanto import is_weight_qbytes_tensor, requantize_with_addmm_delta
from .lora import LoRA
from .pinned_buffer import storage_key
from .slots import ParamSlot, canonical_param_name, iter_param_slots
from .tensor_adapters import DenseAddmmTensorAdapter, select_adapter

logger = logging.getLogger(__name__)

__all__ = ["merge_lora"]

_MergePath = Literal["quanto", "dense"]


@dataclass(slots=True)
class _ParamEntry:
    slot: ParamSlot
    param: nn.Parameter


@dataclass(slots=True)
class _ParamGroup:
    param: nn.Parameter
    storage: tuple[Any, ...]
    entries: list[_ParamEntry]


def merge_lora(
    model: nn.Module,
    loras: Sequence[tuple[LoRA, float]],
) -> int:
    """Merge one or more LoRAs into model parameters in-place.

    Returns the number of parameters that were modified.
    """
    param_index, entries = _build_param_index(model)

    ops: list[
        tuple[str, _ParamGroup, torch.Tensor, torch.Tensor, float, _MergePath]
    ] = []
    groups: dict[tuple[Any, ...], _ParamGroup] = {}
    seen_storage: dict[tuple[Any, ...], str] = {}
    for lora, strength in loras:
        for target_key, (a, b) in lora.targets.items():
            entry = param_index.get(target_key)
            if entry is None:
                continue
            group = _resolve_group(entry, entries, groups)
            existing_target = seen_storage.setdefault(group.storage, target_key)
            if existing_target != target_key:
                raise ValueError(
                    f"LoRA targets {existing_target!r} and {target_key!r} "
                    f"resolve to the same tied parameter storage. Apply "
                    f"only one name for a tied weight in a single "
                    f"merge_lora() call; otherwise the same base weight "
                    f"would receive multiple logical updates."
                )
            expected = tuple(group.param.shape)
            if expected != (b.shape[0], a.shape[1]):
                raise ValueError(
                    f"LoRA factor shape mismatch for {target_key!r}: "
                    f"B@A produces ({b.shape[0]}, {a.shape[1]}), "
                    f"target shape is {expected}."
                )
            data = group.param.data
            merge_path = _merge_path(data, target_key)
            ops.append((target_key, group, a, b, strength, merge_path))

    for target_key, group, a, b, strength, merge_path in ops:
        param = group.param
        if merge_path == "quanto":
            new_qt = requantize_with_addmm_delta(param.data, a, b, strength)
            new_param = nn.Parameter(new_qt, requires_grad=param.requires_grad)
            group.param = _replace_group_param(group, new_param, target_key)
        else:
            dev = param.data.device
            param.data.addmm_(
                b.to(device=dev, dtype=param.data.dtype),
                a.to(device=dev, dtype=param.data.dtype),
                alpha=strength,
            )

    logger.info("merge_lora: merged %d/%d targets", len(ops),
                sum(len(lora.targets) for lora, _ in loras))
    return len(ops)


def _merge_path(data: torch.Tensor, target_key: str) -> _MergePath:
    if is_weight_qbytes_tensor(data):
        return "quanto"

    try:
        adapter = select_adapter(data)
    except NotImplementedError as exc:
        raise ValueError(
            f"Cannot permanently merge LoRA into {target_key!r}: "
            f"tensor type {type(data).__name__} has no registered tensor adapter. "
            "Permanent merge supports plain bf16/fp16/fp32 tensors and "
            "quanto WeightQBytesTensor weights."
        ) from exc

    if not isinstance(adapter, DenseAddmmTensorAdapter):
        raise ValueError(
            f"Cannot permanently merge LoRA into {target_key!r}: "
            f"{adapter.__name__} does not support dense in-place addmm or a "
            "dequantize/requantize merge path. Use routed LoRA for this "
            "tensor type."
        )

    adapter.validate_dense_addmm_target(data, target_key)
    return "dense"


def _build_param_index(model: nn.Module) -> tuple[
    dict[str, _ParamEntry],
    list[_ParamEntry],
]:
    entries: list[_ParamEntry] = []
    index: dict[str, _ParamEntry] = {}
    for slot in iter_param_slots(model):
        entry = _ParamEntry(slot, slot.get())
        entries.append(entry)
        index[canonical_param_name(slot.name)] = entry
    return index, entries


def _resolve_group(
    target: _ParamEntry,
    entries: list[_ParamEntry],
    groups: dict[tuple[Any, ...], _ParamGroup],
) -> _ParamGroup:
    storage = _param_storage_key(target.param)
    group = groups.get(storage)
    if group is not None:
        return group

    group_entries: list[_ParamEntry] = []
    for entry in entries:
        if entry.param is target.param:
            group_entries.append(entry)
            continue
        try:
            entry_storage = _param_storage_key(entry.param)
        except NotImplementedError:
            continue
        if entry_storage == storage:
            group_entries.append(entry)

    group = _ParamGroup(target.param, storage, group_entries)
    groups[storage] = group
    return group


def _param_storage_key(param: nn.Parameter) -> tuple[Any, ...]:
    if param.numel() == 0:
        return ("__empty__", id(param))
    return storage_key(param.data)


def _replace_group_param(
    group: _ParamGroup, new_param: nn.Parameter, target_key: str,
) -> nn.Parameter:
    installed: nn.Parameter | None = None
    seen_slots: set[tuple[int, str]] = set()
    for entry in group.entries:
        slot_key = (id(entry.slot.parent), entry.slot.leaf)
        if slot_key in seen_slots:
            continue
        seen_slots.add(slot_key)
        setattr(entry.slot.parent, entry.slot.leaf, new_param)
        reread = getattr(entry.slot.parent, entry.slot.leaf)
        if not isinstance(reread, nn.Parameter):
            raise RuntimeError(
                f"After merging into quanto target {target_key!r}, reading "
                f"{entry.slot.name!r} from the model returned "
                f"{type(reread).__name__} instead of nn.Parameter. "
                "merge_lora doesn't support parametrize / custom "
                "__setattr__ hooks on quanto-quantized weights."
            )
        if installed is None:
            installed = reread
        elif reread is not installed:
            raise RuntimeError(
                f"After merging into quanto target {target_key!r}, tied "
                "parameter names did not resolve to the same installed Parameter. "
                "merge_lora doesn't support parametrize / custom "
                "__setattr__ hooks that break tied parameter replacement."
            )
    assert installed is not None
    return installed
