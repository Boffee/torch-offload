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

from torch import nn

from ._quanto import QUANTO_AVAILABLE, WeightQBytesTensor, requantize_with_addmm_delta
from .lora import _ADDMM_DTYPES, LoRA
from .slots import canonical_param_name, split_attr_path, walk_attr_path

logger = logging.getLogger(__name__)

__all__ = ["merge_lora"]


def merge_lora(
    model: nn.Module,
    loras: Sequence[tuple[LoRA, float]],
) -> int:
    """Merge one or more LoRAs into model parameters in-place.

    Returns the number of parameters that were modified.
    """
    # Index by canonical key (strips .base_layer. for PEFT), but keep
    # the real name for slot replacement on quanto params.
    param_index: dict[str, tuple[str, nn.Parameter]] = {}
    for name, param in model.named_parameters():
        param_index[canonical_param_name(name)] = (name, param)

    merged = 0
    for lora, strength in loras:
        for target_key, (a, b) in lora.targets.items():
            entry = param_index.get(target_key)
            if entry is None:
                continue
            real_name, param = entry

            if QUANTO_AVAILABLE and isinstance(param.data, WeightQBytesTensor):
                new_qt = requantize_with_addmm_delta(param.data, a, b, strength)
                parent, leaf = split_attr_path(model, real_name)
                setattr(
                    parent, leaf,
                    nn.Parameter(new_qt, requires_grad=param.requires_grad),
                )
                # Slot replacement created a new Parameter — re-read it
                # from the model so subsequent LoRAs targeting the same
                # key see whatever is actually installed (not just the
                # local Parameter we built — a custom __setattr__ /
                # parametrize hook could have transformed it). If the
                # re-read returns a non-Parameter, fail loud rather
                # than silently letting the next iteration mutate a
                # transient object that doesn't reach the model.
                reread = walk_attr_path(model, real_name)
                if not isinstance(reread, nn.Parameter):
                    raise RuntimeError(
                        f"After merging into quanto target "
                        f"{target_key!r}, reading {real_name!r} from "
                        f"the model returned {type(reread).__name__} "
                        "instead of nn.Parameter. merge_lora doesn't "
                        "support parametrize / custom __setattr__ "
                        "hooks on quanto-quantized weights."
                    )
                param_index[target_key] = (real_name, reread)
            elif param.data.dtype in _ADDMM_DTYPES:
                dev = param.data.device
                param.data.addmm_(
                    b.to(device=dev, dtype=param.data.dtype),
                    a.to(device=dev, dtype=param.data.dtype),
                    alpha=strength,
                )
            else:
                raise ValueError(
                    f"Cannot merge LoRA into {target_key!r} with "
                    f"dtype {param.data.dtype}. Supported: "
                    f"bf16/fp16/fp32 or quanto WeightQBytesTensor."
                )
            merged += 1

    logger.info("merge_lora: merged %d/%d targets", merged,
                sum(len(l.targets) for l, _ in loras))
    return merged
