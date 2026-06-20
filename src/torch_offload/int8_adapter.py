"""TorchAO ``Int8Tensor`` adapter (int8 weight-only / int8 dynamic-act).

TorchAO's INT8 workflow stores weights as a tensor subclass with int8
bytes (``qdata``), per-row/per-tensor ``scale``, an optional
``zero_point`` (asymmetric quant), optional static-activation tensors
(``act_quant_scale`` / ``act_quant_zero_point`` / ``act_pre_scale``), and
metadata controlling the matmul dispatch (``block_size``, ``dtype``,
``act_quant_kwargs``). The shared
:class:`~torch_offload.torchao_structured_adapter.TorchaoStructuredAdapter`
base preserves that representation across pinned CPU and GPU storage;
this module supplies the INT8-specific hooks. One adapter covers both
``Int8WeightOnlyConfig`` and ``Int8DynamicActivationInt8WeightConfig``
because TorchAO models them as the same ``Int8Tensor`` parameterized by
``act_quant_kwargs``.

Beyond inference movement, this adapter opts into dequantize/requantize
plus ``copy_into``, enabling merged (permanent) LoRA on int8 bases.
Requantization recomputes the per-block weight scale from the merged
values via the public ``Int8Tensor.from_hp``; like any merge into a
quantized base it is lossy, and int8's 256-level grid makes it lossier
than fp8 — choosing merge vs routed (non-destructive) LoRA is the
caller's tradeoff.

It does not opt into CPU round-trip, trainable ``Parameter.data`` swap,
or activation-scoped dense ``addmm_`` merge: the quant state lives in the
wrapper object, not its bytes, so int8 weights stay frozen for
streaming/training. Routed LoRA remains the non-destructive alternative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ._torchao_int8 import (
    create_int8_tensor,
    dequantize_int8_tensor,
    is_int8_tensor,
    requantize_int8_tensor,
    require_int8_tensor,
    validate_layout,
)
from .tensor_adapters import metadata_key
from .torchao_structured_adapter import TorchaoStructuredAdapter, copy_storage_into


@dataclass(slots=True, frozen=True)
class _Int8Meta:
    """Reconstruction metadata snapshot for a TorchAO Int8 tensor."""

    block_size: tuple[int, ...]
    dtype: torch.dtype  # logical (pre-quantization) dtype
    act_quant_kwargs: object | None


class Int8Adapter(TorchaoStructuredAdapter[_Int8Meta]):
    """Adapter for TorchAO ``Int8Tensor`` weights."""

    _TAG = "torchao-int8"
    _STORAGE_NAMES = (
        "qdata",
        "scale",
        "zero_point",
        "act_quant_scale",
        "act_quant_zero_point",
        "act_pre_scale",
    )

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return is_int8_tensor(t)

    @staticmethod
    def _validate_layout(t: torch.Tensor) -> None:
        validate_layout(t)

    @staticmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_int8_tensor(t)

    @staticmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        return (
            t.qdata,
            t.scale,
            t.zero_point,
            t.act_quant_scale,
            t.act_quant_zero_point,
            t.act_pre_scale,
        )

    @staticmethod
    def _meta_of(t: Any) -> _Int8Meta:  # noqa: ANN401
        return _Int8Meta(
            block_size=tuple(t.block_size),
            dtype=t.dtype,
            act_quant_kwargs=t.act_quant_kwargs,
        )

    @staticmethod
    def _reconstruct(
        storage: tuple[torch.Tensor | None, ...], meta: _Int8Meta
    ) -> torch.Tensor:
        qdata, scale, zero_point, act_quant_scale, act_quant_zero_point, act_pre_scale = (
            storage
        )
        assert qdata is not None
        assert scale is not None
        return create_int8_tensor(
            qdata,
            scale,
            list(meta.block_size),
            meta.dtype,
            zero_point,
            act_quant_scale,
            act_quant_zero_point,
            act_pre_scale,
            meta.act_quant_kwargs,
        )

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        return (
            tuple(t.block_size),
            t.dtype,
            metadata_key(t.act_quant_kwargs),
        )

    @classmethod
    def _layout_metadata(cls, t: Any) -> tuple[object, ...]:  # noqa: ANN401
        # Drop the logical dtype, which layout_signature's standard dtype
        # slot already carries (mirrors Float8Adapter).
        return (
            tuple(t.block_size),
            metadata_key(t.act_quant_kwargs),
        )

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        return t.dtype

    # --- capabilities beyond inference movement ---------------------------

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        return dequantize_int8_tensor(t)

    @staticmethod
    def requantize(t: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        return requantize_int8_tensor(t, like=like)

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        # Fill target's present storage slots (qdata/scale, optional
        # zero_point, and any activation-quant tensors) from the requantized
        # src. Driven by target presence: a symmetric base stored with
        # zero_point=None keeps it, even though from_hp always re-emits a
        # zeros zero_point that carries nothing the target lacks.
        copy_storage_into(
            Int8Adapter._storage_of(require_int8_tensor(src)),
            Int8Adapter._storage_of(require_int8_tensor(target)),
            non_blocking=False,
        )
