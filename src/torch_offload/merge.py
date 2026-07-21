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

import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from torch import nn

from .lora import LoRA, LoRATransform, ScaledLoRAFactor
from .tensor_adapter_registry import param_tensor_id

logger = logging.getLogger(__name__)

__all__ = ["merge_lora"]


@dataclass(slots=True)
class _MergeParamGroup:
    param: nn.Parameter
    tensor_id: tuple[Any, ...]


@dataclass(slots=True)
class _MergeOp:
    group: _MergeParamGroup
    target_key: str
    factors: list[ScaledLoRAFactor]

    @property
    def transform(self) -> LoRATransform:
        return LoRATransform(self.factors)


def merge_lora(
    model: nn.Module,
    loras: Sequence[tuple[LoRA, float]],
) -> int:
    """Merge one or more LoRAs into model parameters in-place.

    Returns the number of unique parameters that were modified. Every LoRA is
    claimed exclusively for the duration of the merge; an adapter already
    attached to another model fails immediately. All target names and merge
    capabilities are validated before any parameter is modified.
    """
    if len({id(lora) for lora, _strength in loras}) != len(loras):
        raise ValueError(
            "merge_lora() does not accept the same LoRA instance more than once"
        )

    with contextlib.ExitStack() as stack:
        for lora, _strength in loras:
            if not isinstance(lora, LoRA):
                raise TypeError("merge_lora() expects LoRA instances")
            lora.activate(mode="merge")
            stack.callback(lora.deactivate)
        return _merge_loras(model, loras)


def _merge_loras(
    model: nn.Module,
    loras: Sequence[tuple[LoRA, float]],
) -> int:
    params_by_target = _collect_params_by_target(model)

    missing_targets = sorted({
        target_key
        for lora, _strength in loras
        for target_key in lora.targets
        if target_key not in params_by_target
    })
    if missing_targets:
        sample = sorted(params_by_target)[:3]
        raise ValueError(
            f"LoRA targets are not parameters in the model: {missing_targets}. "
            "LoRA target keys must match the model's parameter names exactly. "
            f"Sample model parameter keys: {sample} ..."
        )

    param_groups_by_tensor_id: dict[tuple[Any, ...], _MergeParamGroup] = {}
    merge_ops_by_tensor_id: dict[tuple[Any, ...], _MergeOp] = {}
    for lora, strength in loras:
        for target_key, factor in lora.targets.items():
            param = params_by_target[target_key]
            group = _param_group_for_param(
                param,
                param_groups_by_tensor_id,
            )
            op = merge_ops_by_tensor_id.get(group.tensor_id)
            if op is None:
                op = _MergeOp(group, target_key, [])
                merge_ops_by_tensor_id[group.tensor_id] = op
            elif op.target_key != target_key:
                raise ValueError(
                    f"LoRA targets {op.target_key!r} and {target_key!r} "
                    f"resolve to the same tied parameter backing. Apply "
                    f"only one name for a tied weight in a single "
                    f"merge_lora() call; otherwise the same base weight "
                    f"would receive multiple logical updates."
                )
            op.factors.append(factor.scaled(strength))

    merge_ops = list(merge_ops_by_tensor_id.values())

    # Preflight every operation before applying any of them. This catches all
    # expected name, shape, and adapter-capability errors without leaving a
    # permanently half-merged model.
    for op in merge_ops:
        try:
            op.transform.validate_target(op.group.param)
        except ValueError as exc:
            raise ValueError(f"Cannot merge LoRA into {op.target_key!r}: {exc}") from exc

    for op in merge_ops:
        try:
            op.transform.apply(op.group.param)
        except ValueError as exc:
            raise ValueError(f"Cannot merge LoRA into {op.target_key!r}: {exc}") from exc

    logger.info(
        "merge_lora: merged %d unique parameters from %d LoRA targets",
        len(merge_ops),
        sum(len(lora.targets) for lora, _ in loras),
    )
    return len(merge_ops)


def _collect_params_by_target(model: nn.Module) -> dict[str, nn.Parameter]:
    params_by_target: dict[str, nn.Parameter] = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        params_by_target[name] = param
    return params_by_target


def _param_group_for_param(
    target_param: nn.Parameter,
    param_groups_by_tensor_id: dict[tuple[Any, ...], _MergeParamGroup],
) -> _MergeParamGroup:
    tensor_id = param_tensor_id(target_param)
    group = param_groups_by_tensor_id.get(tensor_id)
    if group is not None:
        return group

    group = _MergeParamGroup(target_param, tensor_id)
    param_groups_by_tensor_id[tensor_id] = group
    return group
