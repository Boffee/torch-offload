"""Permanent LoRA merge into model weights.

Merges LoRA deltas directly into model parameters, supporting tensors
whose adapter exposes either dense in-place ``addmm_`` or a
dequantize/requantize plus ``copy_into`` update path. Requantized
merges are lossy but standard practice for permanent LoRA merges into
quantized bases.

This uses the same per-parameter :class:`LoRATransform` as activation
merge; the permanence comes from applying it to the model's resident
parameter instead of a freshly loaded activation parameter.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import nn

from .lora import LoRA, LoRATransform
from .module_names import canonical_param_name
from .tensor_adapter_registry import param_storage_key

logger = logging.getLogger(__name__)

__all__ = ["merge_lora"]


@dataclass(slots=True)
class _MergeParamGroup:
    param: nn.Parameter
    storage: tuple[Any, ...]


@dataclass(slots=True)
class _MergeOp:
    group: _MergeParamGroup
    target_key: str
    transform: LoRATransform


def merge_lora(
    model: nn.Module,
    loras: Sequence[tuple[LoRA, float]],
) -> int:
    """Merge one or more LoRAs into model parameters in-place.

    Returns the number of parameters that were modified.
    """
    params_by_target = _collect_params_by_target(model)

    merge_ops: list[_MergeOp] = []
    param_groups_by_storage: dict[tuple[Any, ...], _MergeParamGroup] = {}
    target_by_storage: dict[tuple[Any, ...], str] = {}
    for lora, strength in loras:
        for target_key, (a, b) in lora.targets.items():
            param = params_by_target.get(target_key)
            if param is None:
                continue
            group = _param_group_for_param(
                param, param_groups_by_storage,
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
            transform = LoRATransform([(a, b, strength)])
            merge_ops.append(_MergeOp(group, target_key, transform))

    for op in merge_ops:
        try:
            op.transform.apply(op.group.param)
        except ValueError as exc:
            raise ValueError(
                f"Cannot merge LoRA into {op.target_key!r}: {exc}"
            ) from exc

    logger.info("merge_lora: merged %d/%d targets", len(merge_ops),
                sum(len(lora.targets) for lora, _ in loras))
    return len(merge_ops)


def _collect_params_by_target(model: nn.Module) -> dict[str, nn.Parameter]:
    params_by_target: dict[str, nn.Parameter] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        params_by_target[canonical_param_name(name)] = param
    return params_by_target


def _param_group_for_param(
    target_param: nn.Parameter,
    param_groups_by_storage: dict[tuple[Any, ...], _MergeParamGroup],
) -> _MergeParamGroup:
    storage = param_storage_key(target_param)
    group = param_groups_by_storage.get(storage)
    if group is not None:
        return group

    group = _MergeParamGroup(target_param, storage)
    param_groups_by_storage[storage] = group
    return group
