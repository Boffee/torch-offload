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
from .slots import canonical_param_name, iter_param_slots
from .tensor_adapters import DenseAddmmTensorAdapter, select_adapter

logger = logging.getLogger(__name__)

__all__ = ["merge_lora"]

_MergePath = Literal["quanto", "dense"]


@dataclass(slots=True)
class _ParamAlias:
    name: str
    param: nn.Parameter
    parent: nn.Module
    leaf: str


@dataclass(slots=True)
class _ParamGroup:
    param: nn.Parameter
    storage: tuple[Any, ...]
    aliases: list[_ParamAlias]


def merge_lora(
    model: nn.Module,
    loras: Sequence[tuple[LoRA, float]],
) -> int:
    """Merge one or more LoRAs into model parameters in-place.

    Returns the number of parameters that were modified.
    """
    param_index, aliases = _build_param_index(model)

    ops: list[
        tuple[str, _ParamGroup, torch.Tensor, torch.Tensor, float, _MergePath]
    ] = []
    groups: dict[tuple[Any, ...], _ParamGroup] = {}
    seen_storage: dict[tuple[Any, ...], str] = {}
    for lora, strength in loras:
        for target_key, (a, b) in lora.targets.items():
            alias = param_index.get(target_key)
            if alias is None:
                continue
            group = _resolve_group(alias, aliases, groups)
            existing_target = seen_storage.setdefault(group.storage, target_key)
            if existing_target != target_key:
                raise ValueError(
                    f"LoRA targets {existing_target!r} and {target_key!r} "
                    f"resolve to the same tied parameter storage. Apply "
                    f"only one alias for a tied weight in a single "
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
    dict[str, _ParamAlias],
    list[_ParamAlias],
]:
    aliases: list[_ParamAlias] = []
    index: dict[str, _ParamAlias] = {}
    for slot in iter_param_slots(model):
        alias = _ParamAlias(slot.name, slot.param, slot.parent, slot.leaf)
        aliases.append(alias)
        index[canonical_param_name(slot.name)] = alias
    return index, aliases


def _resolve_group(
    target: _ParamAlias,
    aliases: list[_ParamAlias],
    groups: dict[tuple[Any, ...], _ParamGroup],
) -> _ParamGroup:
    storage = _param_storage_key(target.param)
    group = groups.get(storage)
    if group is not None:
        return group

    group_aliases: list[_ParamAlias] = []
    for alias in aliases:
        if alias.param is target.param:
            group_aliases.append(alias)
            continue
        try:
            alias_storage = _param_storage_key(alias.param)
        except NotImplementedError:
            continue
        if alias_storage == storage:
            group_aliases.append(alias)

    group = _ParamGroup(target.param, storage, group_aliases)
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
    for alias in group.aliases:
        slot_key = (id(alias.parent), alias.leaf)
        if slot_key in seen_slots:
            continue
        seen_slots.add(slot_key)
        setattr(alias.parent, alias.leaf, new_param)
        reread = getattr(alias.parent, alias.leaf)
        if not isinstance(reread, nn.Parameter):
            raise RuntimeError(
                f"After merging into quanto target {target_key!r}, reading "
                f"{alias.name!r} from the model returned "
                f"{type(reread).__name__} instead of nn.Parameter. "
                "merge_lora doesn't support parametrize / custom "
                "__setattr__ hooks on quanto-quantized weights."
            )
        if installed is None:
            installed = reread
        elif reread is not installed:
            raise RuntimeError(
                f"After merging into quanto target {target_key!r}, tied "
                "aliases did not resolve to the same installed Parameter. "
                "merge_lora doesn't support parametrize / custom "
                "__setattr__ hooks that break tied parameter replacement."
            )
    assert installed is not None
    return installed
