"""TorchAO ``MXTensor`` adapter (OCP microscaling MXFP8 / MXFP4).

TorchAO's MX workflow stores weights as a tensor subclass with packed
element bytes (``qdata`` — ``float8_e4m3fn``/``float8_e5m2`` for MXFP8,
packed ``uint8`` for MXFP4), E8M0 power-of-two block scales (``scale``),
and metadata controlling the matmul dispatch (``elem_dtype``,
``block_size``, ``kernel_preference``, ``act_quant_kwargs``,
``is_swizzled_scales``). The shared
:class:`~torch_offload.torchao_structured_adapter.TorchaoStructuredAdapter`
base preserves that representation across pinned CPU and GPU storage;
this module supplies the MX-specific hooks. One adapter covers both MXFP8
and MXFP4 because TorchAO models them as the same ``MXTensor`` subclass
parameterized by ``elem_dtype``.

Beyond inference movement, this adapter opts into dequantize/requantize
plus ``copy_into``, enabling merged (permanent) LoRA on MX bases.
Requantization re-derives the power-of-two (E8M0) block scales from the
merged values via the public ``MXTensor.to_mx``; like any merge into a
quantized base it is lossy. MXFP4's 4-bit grid makes a requantized merge
far coarser than MXFP8 — choosing merge vs routed (non-destructive) LoRA
is the caller's tradeoff, and both element dtypes support either route.

It does not opt into CPU round-trip or trainable ``Parameter.data`` swap:
the quant state lives in the wrapper object, not its bytes, so MX weights
stay frozen for streaming/training. Routed LoRA remains the
non-destructive alternative when the owning module is a logical
``nn.Linear`` with compatible shape/dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ._torchao_mx import (
    create_mx_tensor,
    dequantize_mx_tensor,
    is_mx_tensor,
    requantize_mx_tensor,
    require_mx_tensor,
    validate_layout,
)
from .tensor_adapters import metadata_key
from .torchao_structured_adapter import TorchaoStructuredAdapter, copy_storage_into


@dataclass(slots=True, frozen=True)
class _MxMeta:
    """Reconstruction metadata snapshot for a TorchAO MX tensor."""

    elem_dtype: object
    block_size: int
    orig_dtype: torch.dtype
    kernel_preference: object
    act_quant_kwargs: object | None
    is_swizzled_scales: bool


class MxAdapter(TorchaoStructuredAdapter[_MxMeta]):
    """Adapter for TorchAO ``MXTensor`` (MXFP8 / MXFP4) weights."""

    _TAG = "torchao-mx"
    _STORAGE_NAMES = ("qdata", "scale")

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return is_mx_tensor(t)

    @staticmethod
    def _validate_layout(t: torch.Tensor) -> None:
        validate_layout(t)

    @staticmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_mx_tensor(t)

    @staticmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        return (t.qdata, t.scale)

    @staticmethod
    def _meta_of(t: Any) -> _MxMeta:  # noqa: ANN401
        return _MxMeta(
            elem_dtype=t.elem_dtype,
            block_size=t.block_size,
            orig_dtype=t.orig_dtype,
            kernel_preference=t.kernel_preference,
            act_quant_kwargs=t.act_quant_kwargs,
            is_swizzled_scales=t.is_swizzled_scales,
        )

    @staticmethod
    def _reconstruct(
        storage: tuple[torch.Tensor | None, ...], meta: _MxMeta
    ) -> torch.Tensor:
        qdata, scale = storage
        assert qdata is not None
        assert scale is not None
        return create_mx_tensor(
            qdata,
            scale,
            meta.elem_dtype,
            meta.block_size,
            meta.orig_dtype,
            meta.kernel_preference,
            meta.act_quant_kwargs,
            meta.is_swizzled_scales,
        )

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        return (
            t.elem_dtype,
            t.block_size,
            t.orig_dtype,
            t.is_swizzled_scales,
            metadata_key(t.kernel_preference),
            metadata_key(t.act_quant_kwargs),
        )

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        return t.orig_dtype

    # --- capabilities beyond inference movement ---------------------------

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        return dequantize_mx_tensor(t)

    @staticmethod
    def requantize(t: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        return requantize_mx_tensor(t, like=like)

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        # Copy the packed elements + E8M0 block scales into target's
        # existing buffers, preserving its wrapper/object identity. MX has
        # no optional storage, so both slots are always present.
        copy_storage_into(
            MxAdapter._storage_of(require_mx_tensor(src)),
            MxAdapter._storage_of(require_mx_tensor(target)),
            non_blocking=False,
        )
