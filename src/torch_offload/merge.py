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

import torch
from torch import nn

from .lora import LoRA

logger = logging.getLogger(__name__)

__all__ = ["merge_lora"]

_ADDMM_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

try:
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

    _QUANTO_AVAILABLE = True
except ImportError:
    _QUANTO_AVAILABLE = False
    WeightQBytesTensor = None  # type: ignore[assignment,misc]


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
        canonical = name.replace(".base_layer.", ".")
        param_index[canonical] = (name, param)

    merged = 0
    for lora, strength in loras:
        for target_key, (a, b) in lora.targets.items():
            entry = param_index.get(target_key)
            if entry is None:
                continue
            real_name, param = entry

            if _QUANTO_AVAILABLE and isinstance(param.data, WeightQBytesTensor):
                _merge_quanto(model, real_name, param, a, b, strength)
                # Slot replacement created a new Parameter — update the
                # index so subsequent LoRAs targeting the same key see it.
                param_index[target_key] = (real_name, _get_param(model, real_name))
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


def _merge_quanto(
    model: nn.Module,
    real_name: str,
    param: nn.Parameter,
    a: torch.Tensor,
    b: torch.Tensor,
    strength: float,
) -> None:
    qt = param.data
    dev = qt.device
    float_data = qt.dequantize().to(device=dev, dtype=torch.float32)
    float_data.addmm_(
        b.to(device=dev, dtype=torch.float32),
        a.to(device=dev, dtype=torch.float32),
        alpha=strength,
    )
    new_qt = WeightQBytesTensor.create(
        qt.qtype, qt.axis, qt.size(), qt.stride(),
        _quantize_to_qbytes(float_data, qt),
        qt._scale.clone(),
        getattr(qt, "activation_qtype", None),
    )
    _replace_param(model, real_name, nn.Parameter(new_qt, requires_grad=param.requires_grad))


def _quantize_to_qbytes(
    float_data: torch.Tensor, reference: "WeightQBytesTensor",
) -> torch.Tensor:
    """Quantize float data using the same scale as the reference tensor."""
    scale = reference._scale
    axis = reference.axis
    if axis == 0:
        scaled = float_data / scale.view(-1, *([1] * (float_data.dim() - 1)))
    else:
        scaled = float_data / scale
    return scaled.round().clamp(
        torch.iinfo(reference._data.dtype).min,
        torch.iinfo(reference._data.dtype).max,
    ).to(reference._data.dtype)


def _get_param(model: nn.Module, dotted_key: str) -> nn.Parameter:
    obj = model
    for part in dotted_key.split("."):
        obj = getattr(obj, part)
    return obj  # type: ignore[return-value]


def _replace_param(model: nn.Module, dotted_key: str, new_param: nn.Parameter) -> None:
    parts = dotted_key.rsplit(".", 1)
    if len(parts) == 1:
        raise ValueError(f"Cannot replace top-level parameter {dotted_key!r}")
    parent_path, leaf = parts
    parent = model
    for part in parent_path.split("."):
        parent = getattr(parent, part)
    setattr(parent, leaf, new_param)
