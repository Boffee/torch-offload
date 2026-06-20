"""TorchAO ``NVFP4Tensor`` adapter.

TorchAO's NVFP4 workflow stores weights as a tensor subclass with packed
FP4 bytes (``qdata``), FP8 block scales (``scale``), optional global
per-tensor scales (``per_tensor_scale`` / ``act_per_tensor_scale``), and
metadata controlling the matmul dispatch. The shared
:class:`~torch_offload.torchao_structured_adapter.TorchaoStructuredAdapter`
base preserves that representation across pinned CPU and GPU storage;
this module supplies the NVFP4-specific hooks. The two global scales are
optional, represented as ``None`` entries in the storage tuple so the
base's clone/alloc/copy/accounting skip them.

Beyond inference movement, this adapter opts into dequantize/requantize
plus ``copy_into``, enabling merged (permanent) LoRA on NVFP4 bases.
Requantization re-derives the FP8 (E4M3) block scales — and, for
two-level scaling, the global ``per_tensor_scale`` — from the merged
values via the public ``NVFP4Tensor.to_nvfp4``; like any merge into a
quantized base it is lossy, and NVFP4's 4-bit grid makes it coarse, so
choosing merge vs routed (non-destructive) LoRA is the caller's tradeoff.

It does not opt into CPU round-trip or trainable ``Parameter.data`` swap:
the quant state lives in the wrapper object, not its bytes, so NVFP4
weights stay frozen for streaming/training. Routed LoRA remains the
non-destructive alternative when the owning module is a logical
``nn.Linear`` with compatible shape/dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ._torchao_nvfp4 import (
    create_nvfp4_tensor,
    dequantize_nvfp4_tensor,
    is_nvfp4_tensor,
    requantize_nvfp4_tensor,
    require_nvfp4_tensor,
    validate_layout,
)
from .tensor_adapters import metadata_key
from .torchao_structured_adapter import TorchaoStructuredAdapter, copy_storage_into


@dataclass(slots=True, frozen=True)
class _Nvfp4Meta:
    """Reconstruction metadata snapshot for a TorchAO NVFP4 tensor."""

    block_size: int
    orig_dtype: torch.dtype
    is_swizzled_scales: bool
    use_triton_kernel: bool
    act_quant_kwargs: object | None


class Nvfp4Adapter(TorchaoStructuredAdapter[_Nvfp4Meta]):
    """Adapter for TorchAO ``NVFP4Tensor`` weights."""

    _TAG = "torchao-nvfp4"
    _STORAGE_NAMES = ("qdata", "scale", "per_tensor_scale", "act_per_tensor_scale")

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return is_nvfp4_tensor(t)

    @staticmethod
    def _validate_layout(t: torch.Tensor) -> None:
        validate_layout(t)

    @staticmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_nvfp4_tensor(t)

    @staticmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        return (t.qdata, t.scale, t.per_tensor_scale, t.act_per_tensor_scale)

    @staticmethod
    def _meta_of(t: Any) -> _Nvfp4Meta:  # noqa: ANN401
        return _Nvfp4Meta(
            block_size=t.block_size,
            orig_dtype=t.orig_dtype,
            is_swizzled_scales=t.is_swizzled_scales,
            use_triton_kernel=t.use_triton_kernel,
            act_quant_kwargs=t.act_quant_kwargs,
        )

    @staticmethod
    def _reconstruct(
        storage: tuple[torch.Tensor | None, ...], meta: _Nvfp4Meta
    ) -> torch.Tensor:
        qdata, scale, per_tensor_scale, act_per_tensor_scale = storage
        assert qdata is not None
        assert scale is not None
        return create_nvfp4_tensor(
            qdata,
            scale,
            meta.block_size,
            meta.orig_dtype,
            per_tensor_scale,
            act_per_tensor_scale,
            meta.is_swizzled_scales,
            meta.use_triton_kernel,
            meta.act_quant_kwargs,
        )

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        return (
            t.block_size,
            t.orig_dtype,
            t.is_swizzled_scales,
            t.use_triton_kernel,
            metadata_key(t.act_quant_kwargs),
        )

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        return t.orig_dtype

    # --- capabilities beyond inference movement ---------------------------

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        return dequantize_nvfp4_tensor(t)

    @staticmethod
    def requantize(t: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        return requantize_nvfp4_tensor(t, like=like)

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        # Fill target's present storage slots (packed FP4 + E4M3 block
        # scales, plus the optional global per-tensor/activation scales)
        # from the requantized src, preserving target's wrapper identity.
        copy_storage_into(
            Nvfp4Adapter._storage_of(require_nvfp4_tensor(src)),
            Nvfp4Adapter._storage_of(require_nvfp4_tensor(target)),
            non_blocking=False,
        )
