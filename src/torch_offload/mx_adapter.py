"""TorchAO ``MXTensor`` adapter (OCP microscaling MXFP8 / MXFP4).

TorchAO's MX workflow stores weights as a tensor subclass with packed
element bytes (``qdata`` — ``float8_e4m3fn``/``float8_e5m2`` for MXFP8,
packed ``uint8`` for MXFP4), E8M0 power-of-two block scales (``scale``),
and metadata controlling the matmul dispatch (``elem_dtype``,
``block_size``, ``kernel_preference``, ``act_quant_kwargs``,
``is_swizzled_scales``). This adapter preserves that representation
across pinned CPU storage and GPU storage. One adapter covers both MXFP8
and MXFP4 because TorchAO models them as the same ``MXTensor`` subclass
parameterized by ``elem_dtype``.

The adapter intentionally exposes inference movement only. MX model
weights are treated as frozen: no CPU round-trip, no trainable
``Parameter.data`` swap, and no activation-scoped dense ``addmm_`` LoRA
merge. Routed LoRA remains possible when the owning module is a logical
``nn.Linear`` with compatible shape/dtype.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ._torchao_mx import (
    create_mx_tensor,
    is_mx_tensor,
    require_mx_tensor,
    validate_layout,
)
from .tensor_adapters import (
    clone_to_pinned_cpu,
    empty_like_strided,
    metadata_key,
    optional_tensor_id,
    tensor_layout,
)


@dataclass(slots=True)
class _MxPinned:
    """Pinned-CPU state for a TorchAO MX tensor."""

    qdata: torch.Tensor
    scale: torch.Tensor
    elem_dtype: object
    block_size: int
    orig_dtype: torch.dtype
    kernel_preference: object
    act_quant_kwargs: object | None
    is_swizzled_scales: bool


@dataclass(slots=True)
class _MxGpu:
    """GPU state for a TorchAO MX tensor: storage only; quant metadata
    lives in the originating :class:`_MxPinned`."""

    qdata: torch.Tensor
    scale: torch.Tensor


def _build_mx(
    state: _MxPinned, qdata: torch.Tensor, scale: torch.Tensor,
) -> torch.Tensor:
    return create_mx_tensor(
        qdata,
        scale,
        state.elem_dtype,
        state.block_size,
        state.orig_dtype,
        state.kernel_preference,
        state.act_quant_kwargs,
        state.is_swizzled_scales,
    )


class MxAdapter:
    """Adapter for TorchAO ``MXTensor`` (MXFP8 / MXFP4) weights."""

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        if not is_mx_tensor(t):
            return False
        validate_layout(t)
        return True

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple[object, ...]:
        mx = require_mx_tensor(t)
        return (
            "torchao-mx",
            optional_tensor_id(mx.qdata),
            optional_tensor_id(mx.scale),
            tuple(mx.shape),
            mx.stride(),
            mx.elem_dtype,
            mx.block_size,
            mx.orig_dtype,
            mx.is_swizzled_scales,
            metadata_key(mx.kernel_preference),
            metadata_key(mx.act_quant_kwargs),
        )

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple[object, ...]:
        """Hashable layout metadata used by block-pool compatibility checks."""
        mx = require_mx_tensor(t)
        return (
            tuple(mx.shape),
            mx.dtype,
            mx.stride(),
            mx.elem_dtype,
            mx.block_size,
            mx.orig_dtype,
            mx.is_swizzled_scales,
            metadata_key(mx.kernel_preference),
            metadata_key(mx.act_quant_kwargs),
            ("qdata", tensor_layout(mx.qdata)),
            ("scale", tensor_layout(mx.scale)),
        )

    @staticmethod
    def clone_pin(t: torch.Tensor) -> _MxPinned:
        mx = require_mx_tensor(t)
        return _MxPinned(
            # preserve_format: qdata/scale stride ordering can encode a
            # transposed MX tensor, mirroring the NVFP4 handling.
            qdata=clone_to_pinned_cpu(mx.qdata),
            scale=clone_to_pinned_cpu(mx.scale),
            elem_dtype=mx.elem_dtype,
            block_size=mx.block_size,
            orig_dtype=mx.orig_dtype,
            kernel_preference=mx.kernel_preference,
            act_quant_kwargs=mx.act_quant_kwargs,
            is_swizzled_scales=mx.is_swizzled_scales,
        )

    @staticmethod
    def cpu_param(
        state: _MxPinned, *, requires_grad: bool = False
    ) -> nn.Parameter:
        return nn.Parameter(
            _build_mx(state, state.qdata, state.scale),
            requires_grad=requires_grad,
        )

    @staticmethod
    def alloc_gpu(state: _MxPinned, device: torch.device) -> _MxGpu:
        return _MxGpu(
            qdata=empty_like_strided(state.qdata, device),
            scale=empty_like_strided(state.scale, device),
        )

    @staticmethod
    def gpu_param(
        pinned: _MxPinned,
        gpu_state: _MxGpu,
        *,
        requires_grad: bool = False,
    ) -> nn.Parameter:
        return nn.Parameter(
            _build_mx(pinned, gpu_state.qdata, gpu_state.scale),
            requires_grad=requires_grad,
        )

    @staticmethod
    def copy_to_gpu(
        src: _MxPinned, dst: _MxGpu, *, non_blocking: bool = False
    ) -> None:
        dst.qdata.copy_(src.qdata, non_blocking=non_blocking)
        dst.scale.copy_(src.scale, non_blocking=non_blocking)

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        mx = require_mx_tensor(t)
        return mx.orig_dtype

    @staticmethod
    def cache_bytes(state: _MxPinned) -> int:
        return state.qdata.nbytes + state.scale.nbytes
