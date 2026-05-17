"""TorchAO ``NVFP4Tensor`` adapter.

TorchAO's NVFP4 workflow stores weights as a tensor subclass with
packed FP4 bytes (``qdata``), FP8 block scales (``scale``), optional
global per-tensor scales, and metadata controlling the matmul dispatch.
This adapter preserves that representation across pinned CPU storage and
GPU slot storage.

The adapter intentionally exposes inference movement only. NVFP4 model
weights are treated as frozen: no CPU round-trip, no trainable
``Parameter.data`` swap, and no activation-scoped dense ``addmm_`` LoRA
merge. Routed LoRA remains possible when the owning module is a logical
``nn.Linear`` with compatible shape/dtype.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass

import torch
from torch import nn

from ._torchao_nvfp4 import (
    TORCHAO_NVFP4_AVAILABLE,
    create_nvfp4_tensor,
    is_nvfp4_tensor,
    require_nvfp4_tensor,
    validate_layout,
)
from .tensor_adapters import register_adapter


@dataclass(slots=True)
class _Nvfp4Pinned:
    """Pinned-CPU state for a TorchAO NVFP4 tensor."""

    qdata: torch.Tensor
    scale: torch.Tensor
    block_size: int
    orig_dtype: torch.dtype
    per_tensor_scale: torch.Tensor | None
    act_per_tensor_scale: torch.Tensor | None
    is_swizzled_scales: bool
    use_triton_kernel: bool
    act_quant_kwargs: object | None


@dataclass(slots=True)
class _Nvfp4Gpu:
    """GPU state for a TorchAO NVFP4 tensor."""

    qdata: torch.Tensor
    scale: torch.Tensor
    per_tensor_scale: torch.Tensor | None
    act_per_tensor_scale: torch.Tensor | None


def _clone_pin(t: torch.Tensor) -> torch.Tensor:
    # Preserve qdata/scale strides: TorchAO uses qdata stride ordering to
    # represent transposed NVFP4 tensors.
    return t.clone().pin_memory()


def _empty_like_strided(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.empty_strided(
        tuple(t.shape),
        t.stride(),
        dtype=t.dtype,
        device=device,
    )


def _tensor_storage_key(t: torch.Tensor | None) -> tuple[object, ...] | None:
    if t is None:
        return None
    return (
        t.data_ptr(),
        t.dtype,
        tuple(t.shape),
        t.stride(),
        t.storage_offset(),
    )


def _tensor_layout(t: torch.Tensor | None) -> tuple[object, ...] | None:
    if t is None:
        return None
    return (tuple(t.shape), t.dtype, t.stride())


def _metadata_key(value: object | None) -> object | None:
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__module__,
            type(value).__qualname__,
            _make_hashable(asdict(value)),
        )
    return repr(value)


def _make_hashable(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(
            (repr(k), _make_hashable(v))
            for k, v in sorted(value.items(), key=lambda item: repr(item[0]))
        )
    if isinstance(value, (tuple, list)):
        return tuple(_make_hashable(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted((_make_hashable(v) for v in value), key=repr))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _build_nvfp4(
    state: _Nvfp4Pinned,
    qdata: torch.Tensor,
    scale: torch.Tensor,
    per_tensor_scale: torch.Tensor | None,
    act_per_tensor_scale: torch.Tensor | None,
) -> torch.Tensor:
    return create_nvfp4_tensor(
        qdata,
        scale,
        state.block_size,
        state.orig_dtype,
        per_tensor_scale,
        act_per_tensor_scale,
        state.is_swizzled_scales,
        state.use_triton_kernel,
        state.act_quant_kwargs,
    )


class Nvfp4Adapter:
    """Adapter for TorchAO ``NVFP4Tensor`` weights."""

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        if not is_nvfp4_tensor(t):
            return False
        validate_layout(t)
        return True

    @staticmethod
    def storage_key(t: torch.Tensor) -> tuple[object, ...]:
        qt = require_nvfp4_tensor(t)
        return (
            "torchao-nvfp4",
            _tensor_storage_key(qt.qdata),
            _tensor_storage_key(qt.scale),
            _tensor_storage_key(qt.per_tensor_scale),
            _tensor_storage_key(qt.act_per_tensor_scale),
            tuple(qt.shape),
            qt.stride(),
            qt.block_size,
            qt.orig_dtype,
            qt.is_swizzled_scales,
            qt.use_triton_kernel,
            _metadata_key(qt.act_quant_kwargs),
        )

    @staticmethod
    def clone_pin(t: torch.Tensor) -> _Nvfp4Pinned:
        qt = require_nvfp4_tensor(t)
        return _Nvfp4Pinned(
            qdata=_clone_pin(qt.qdata),
            scale=_clone_pin(qt.scale),
            block_size=qt.block_size,
            orig_dtype=qt.orig_dtype,
            per_tensor_scale=(
                _clone_pin(qt.per_tensor_scale)
                if qt.per_tensor_scale is not None
                else None
            ),
            act_per_tensor_scale=(
                _clone_pin(qt.act_per_tensor_scale)
                if qt.act_per_tensor_scale is not None
                else None
            ),
            is_swizzled_scales=qt.is_swizzled_scales,
            use_triton_kernel=qt.use_triton_kernel,
            act_quant_kwargs=qt.act_quant_kwargs,
        )

    @staticmethod
    def cpu_param(
        state: _Nvfp4Pinned, *, requires_grad: bool = False
    ) -> nn.Parameter:
        return nn.Parameter(
            _build_nvfp4(
                state,
                state.qdata,
                state.scale,
                state.per_tensor_scale,
                state.act_per_tensor_scale,
            ),
            requires_grad=requires_grad,
        )

    @staticmethod
    def alloc_gpu(state: _Nvfp4Pinned, device: torch.device) -> _Nvfp4Gpu:
        return _Nvfp4Gpu(
            qdata=_empty_like_strided(state.qdata, device),
            scale=_empty_like_strided(state.scale, device),
            per_tensor_scale=(
                _empty_like_strided(state.per_tensor_scale, device)
                if state.per_tensor_scale is not None
                else None
            ),
            act_per_tensor_scale=(
                _empty_like_strided(state.act_per_tensor_scale, device)
                if state.act_per_tensor_scale is not None
                else None
            ),
        )

    @staticmethod
    def gpu_param(
        pinned: _Nvfp4Pinned,
        gpu_state: _Nvfp4Gpu,
        *,
        requires_grad: bool = False,
    ) -> nn.Parameter:
        return nn.Parameter(
            _build_nvfp4(
                pinned,
                gpu_state.qdata,
                gpu_state.scale,
                gpu_state.per_tensor_scale,
                gpu_state.act_per_tensor_scale,
            ),
            requires_grad=requires_grad,
        )

    @staticmethod
    def copy_to_gpu(
        src: _Nvfp4Pinned, dst: _Nvfp4Gpu, *, non_blocking: bool = False
    ) -> None:
        dst.qdata.copy_(src.qdata, non_blocking=non_blocking)
        dst.scale.copy_(src.scale, non_blocking=non_blocking)
        if src.per_tensor_scale is not None:
            assert dst.per_tensor_scale is not None
            dst.per_tensor_scale.copy_(
                src.per_tensor_scale, non_blocking=non_blocking
            )
        if src.act_per_tensor_scale is not None:
            assert dst.act_per_tensor_scale is not None
            dst.act_per_tensor_scale.copy_(
                src.act_per_tensor_scale, non_blocking=non_blocking
            )

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        qt = require_nvfp4_tensor(t)
        return qt.orig_dtype

    @staticmethod
    def cache_bytes(state: _Nvfp4Pinned) -> int:
        total = state.qdata.nbytes + state.scale.nbytes
        if state.per_tensor_scale is not None:
            total += state.per_tensor_scale.nbytes
        if state.act_per_tensor_scale is not None:
            total += state.act_per_tensor_scale.nbytes
        return total

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple[object, ...]:
        """Hashable layout metadata used by block-pool compatibility checks."""
        qt = require_nvfp4_tensor(t)
        return (
            tuple(qt.shape),
            qt.dtype,
            qt.stride(),
            qt.block_size,
            qt.orig_dtype,
            qt.is_swizzled_scales,
            qt.use_triton_kernel,
            _metadata_key(qt.act_quant_kwargs),
            ("qdata", _tensor_layout(qt.qdata)),
            ("scale", _tensor_layout(qt.scale)),
            ("per_tensor_scale", _tensor_layout(qt.per_tensor_scale)),
            ("act_per_tensor_scale", _tensor_layout(qt.act_per_tensor_scale)),
        )


if TORCHAO_NVFP4_AVAILABLE:
    register_adapter(Nvfp4Adapter)
