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
from .pinned_param import storage_key
from .slots import ParamSlot, canonical_param_name, iter_param_slots, unique_slots
from .tensor_adapters import DenseAddmmTensorAdapter, select_adapter

logger = logging.getLogger(__name__)

__all__ = ["merge_lora"]

_MergePath = Literal["quanto", "dense"]


@dataclass(slots=True)
class _MergeParamGroup:
    param: nn.Parameter
    storage: tuple[Any, ...]
    slots: list[ParamSlot]


def merge_lora(
    model: nn.Module,
    loras: Sequence[tuple[LoRA, float]],
) -> int:
    """Merge one or more LoRAs into model parameters in-place.

    Returns the number of parameters that were modified.
    """
    param_index, slots = _build_param_index(model)

    ops: list[
        tuple[_MergeParamGroup, torch.Tensor, torch.Tensor, float, _MergePath]
    ] = []
    groups: dict[tuple[Any, ...], _MergeParamGroup] = {}
    seen_storage: dict[tuple[Any, ...], str] = {}
    for lora, strength in loras:
        for target_key, (a, b) in lora.targets.items():
            slot = param_index.get(target_key)
            if slot is None:
                continue
            group = _resolve_group(slot, slots, groups)
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
            ops.append((group, a, b, strength, merge_path))

    for group, a, b, strength, merge_path in ops:
        param = group.param
        if merge_path == "quanto":
            new_qt = requantize_with_addmm_delta(param.data, a, b, strength)
            new_param = nn.Parameter(new_qt, requires_grad=param.requires_grad)
            group.param = _replace_group_param(group, new_param)
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
    dict[str, ParamSlot],
    list[ParamSlot],
]:
    slots: list[ParamSlot] = []
    index: dict[str, ParamSlot] = {}
    for slot in iter_param_slots(model):
        slots.append(slot)
        index[canonical_param_name(slot.name)] = slot
    return index, slots


def _resolve_group(
    target: ParamSlot,
    slots: list[ParamSlot],
    groups: dict[tuple[Any, ...], _MergeParamGroup],
) -> _MergeParamGroup:
    target_param = target.get()
    storage = _param_storage_key(target_param)
    group = groups.get(storage)
    if group is not None:
        return group

    group_slots: list[ParamSlot] = []
    for slot in slots:
        param = slot.get()
        if param is target_param:
            group_slots.append(slot)
            continue
        try:
            slot_storage = _param_storage_key(param)
        except NotImplementedError:
            continue
        if slot_storage == storage:
            group_slots.append(slot)

    group = _MergeParamGroup(target_param, storage, group_slots)
    groups[storage] = group
    return group


def _param_storage_key(param: nn.Parameter) -> tuple[Any, ...]:
    if param.numel() == 0:
        return ("__empty__", id(param))
    return storage_key(param.data)


def _replace_group_param(
    group: _MergeParamGroup, new_param: nn.Parameter,
) -> nn.Parameter:
    for slot in unique_slots(group.slots):
        slot.set(new_param)
    return new_param
