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

The adapter exposes inference movement only. NVFP4 model weights are
treated as frozen: no CPU round-trip, no trainable ``Parameter.data``
swap, and no activation-scoped dense ``addmm_`` LoRA merge. Routed LoRA
remains possible when the owning module is a logical ``nn.Linear`` with
compatible shape/dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ._torchao_nvfp4 import (
    create_nvfp4_tensor,
    is_nvfp4_tensor,
    require_nvfp4_tensor,
    validate_layout,
)
from .tensor_adapters import metadata_key
from .torchao_structured_adapter import TorchaoStructuredAdapter


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
