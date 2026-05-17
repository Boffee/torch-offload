"""Permanent LoRA merge into model weights.

Merges LoRA deltas directly into model parameters, supporting tensors
whose adapter exposes either dense in-place ``addmm_`` or a
dequantize/requantize update path. Requantized merges are lossy but
standard practice for permanent LoRA merges into quantized bases.

Unlike :class:`LoRATransform` (which merges at DMA time and is
reversible), this is a one-shot permanent modification.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn

from .lora import LoRA
from .pinned_param import storage_key
from .slots import ParamSlot, canonical_param_name, iter_param_slots, unique_slots
from .tensor_adapters import (
    DenseAddmmTensorAdapter,
    DequantRequantTensorAdapter,
    select_adapter,
)

logger = logging.getLogger(__name__)

__all__ = ["merge_lora"]

_DequantRequantAdapter = type[DequantRequantTensorAdapter[Any, Any]]


@dataclass(slots=True)
class _MergeParamGroup:
    param: nn.Parameter
    storage: tuple[Any, ...]
    slots: list[ParamSlot]


@dataclass(slots=True)
class _MergeOp:
    group: _MergeParamGroup
    a: torch.Tensor
    b: torch.Tensor
    strength: float
    dequant_requant_adapter: _DequantRequantAdapter | None


def merge_lora(
    model: nn.Module,
    loras: Sequence[tuple[LoRA, float]],
) -> int:
    """Merge one or more LoRAs into model parameters in-place.

    Returns the number of parameters that were modified.
    """
    param_slots_by_target, param_slots = _collect_param_slots(model)

    merge_ops: list[_MergeOp] = []
    param_groups_by_storage: dict[tuple[Any, ...], _MergeParamGroup] = {}
    target_by_storage: dict[tuple[Any, ...], str] = {}
    for lora, strength in loras:
        for target_key, (a, b) in lora.targets.items():
            slot = param_slots_by_target.get(target_key)
            if slot is None:
                continue
            group = _param_group_for_slot(
                slot, param_slots, param_groups_by_storage,
            )
            existing_target = target_by_storage.setdefault(
                group.storage, target_key,
            )
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
            dequant_requant_adapter = _dequant_requant_adapter(data, target_key)
            merge_ops.append(_MergeOp(group, a, b, strength, dequant_requant_adapter))

    for op in merge_ops:
        param = op.group.param
        if op.dequant_requant_adapter is None:
            dev = param.data.device
            param.data.addmm_(
                op.b.to(device=dev, dtype=param.data.dtype),
                op.a.to(device=dev, dtype=param.data.dtype),
                alpha=op.strength,
            )
        else:
            dense = op.dequant_requant_adapter.dequantize(param.data)
            dev = dense.device
            dense.addmm_(
                op.b.to(device=dev, dtype=dense.dtype),
                op.a.to(device=dev, dtype=dense.dtype),
                alpha=op.strength,
            )
            new_data = op.dequant_requant_adapter.requantize(dense, like=param.data)
            new_param = nn.Parameter(new_data, requires_grad=param.requires_grad)
            op.group.param = _replace_group_param(op.group, new_param)

    logger.info("merge_lora: merged %d/%d targets", len(merge_ops),
                sum(len(lora.targets) for lora, _ in loras))
    return len(merge_ops)


def _dequant_requant_adapter(
    data: torch.Tensor, target_key: str,
) -> _DequantRequantAdapter | None:
    try:
        adapter = select_adapter(data)
    except NotImplementedError as exc:
        raise ValueError(
            f"Cannot permanently merge LoRA into {target_key!r}: "
            f"tensor type {type(data).__name__} has no registered tensor adapter. "
            "Permanent merge requires a tensor adapter with dense addmm or "
            "dequantize/requantize support."
        ) from exc

    if isinstance(adapter, DenseAddmmTensorAdapter):
        try:
            adapter.validate_dense_addmm_target(data, target_key)
            return None
        except ValueError:
            if isinstance(adapter, DequantRequantTensorAdapter):
                return cast(_DequantRequantAdapter, adapter)
            raise

    if isinstance(adapter, DequantRequantTensorAdapter):
        return cast(_DequantRequantAdapter, adapter)

    raise ValueError(
        f"Cannot permanently merge LoRA into {target_key!r}: "
        f"{adapter.__name__} does not support dense in-place addmm or "
        "dequantize/requantize updates. Use routed LoRA for this tensor type."
    )


def _collect_param_slots(model: nn.Module) -> tuple[
    dict[str, ParamSlot],
    list[ParamSlot],
]:
    param_slots: list[ParamSlot] = []
    param_slots_by_target: dict[str, ParamSlot] = {}
    for slot in iter_param_slots(model):
        param_slots.append(slot)
        param_slots_by_target[canonical_param_name(slot.name)] = slot
    return param_slots_by_target, param_slots


def _param_group_for_slot(
    target_slot: ParamSlot,
    param_slots: list[ParamSlot],
    param_groups_by_storage: dict[tuple[Any, ...], _MergeParamGroup],
) -> _MergeParamGroup:
    target_param = target_slot.get()
    storage = _param_storage_key(target_param)
    group = param_groups_by_storage.get(storage)
    if group is not None:
        return group

    group_slots: list[ParamSlot] = []
    for slot in param_slots:
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
    param_groups_by_storage[storage] = group
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
