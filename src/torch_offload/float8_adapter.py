"""TorchAO ``Float8Tensor`` (scaled-fp8) adapter.

TorchAO's scaled-fp8 workflow stores weights as a tensor subclass with
fp8 bytes (``qdata``), per-row or per-tensor fp32 scales (``scale``),
and metadata controlling the matmul dispatch (``block_size``,
``mm_config``, ``kernel_preference``, ``act_quant_kwargs``). This
adapter preserves that representation across pinned CPU storage and
GPU storage.

Beyond inference movement, the adapter exposes:

- CPU round-trip: GPU storage is the identical fp8 bytes, so D2H back
  into the pinned host state is lossless.
- Dequantize/requantize plus ``copy_into``: enables merged LoRA on
  scaled-fp8 bases. Requantization recomputes scales via the public
  ``Float8Tensor.from_hp``, which is lossy but standard practice for
  permanent merges into quantized weights.

No trainable ``Parameter.data`` swap — the quant state lives in the
wrapper object, not its bytes, so scaled-fp8 weights stay frozen.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ._torchao_float8 import (
    create_float8_tensor,
    dequantize_float8_tensor,
    is_float8_tensor,
    requantize_float8_tensor,
    require_float8_tensor,
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
class _Float8Pinned:
    """Pinned-CPU state for a TorchAO scaled-fp8 tensor."""

    qdata: torch.Tensor
    scale: torch.Tensor
    block_size: list[int]
    mm_config: object | None
    act_quant_kwargs: object | None
    kernel_preference: object
    dtype: torch.dtype  # logical (pre-quantization) dtype


@dataclass(slots=True)
class _Float8Gpu:
    """GPU state for a TorchAO scaled-fp8 tensor: storage only; quant
    metadata lives in the originating :class:`_Float8Pinned`."""

    qdata: torch.Tensor
    scale: torch.Tensor


def _build_float8(
    state: _Float8Pinned, qdata: torch.Tensor, scale: torch.Tensor,
) -> torch.Tensor:
    return create_float8_tensor(
        qdata,
        scale,
        list(state.block_size),
        state.mm_config,
        state.act_quant_kwargs,
        state.kernel_preference,
        state.dtype,
    )


class Float8Adapter:
    """Adapter for TorchAO ``Float8Tensor`` (scaled-fp8) weights."""

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        if not is_float8_tensor(t):
            return False
        validate_layout(t)
        return True

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple[object, ...]:
        f8 = require_float8_tensor(t)
        return (
            "torchao-float8",
            optional_tensor_id(f8.qdata),
            optional_tensor_id(f8.scale),
            tuple(f8.shape),
            f8.stride(),
            tuple(f8.block_size),
            f8.dtype,
            metadata_key(f8.mm_config),
            metadata_key(f8.kernel_preference),
            metadata_key(f8.act_quant_kwargs),
        )

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple[object, ...]:
        """Hashable layout metadata used by block-pool compatibility checks."""
        f8 = require_float8_tensor(t)
        return (
            tuple(f8.shape),
            f8.dtype,
            f8.stride(),
            tuple(f8.block_size),
            metadata_key(f8.mm_config),
            metadata_key(f8.kernel_preference),
            metadata_key(f8.act_quant_kwargs),
            ("qdata", tensor_layout(f8.qdata)),
            ("scale", tensor_layout(f8.scale)),
        )

    @staticmethod
    def clone_pin(t: torch.Tensor) -> _Float8Pinned:
        f8 = require_float8_tensor(t)
        return _Float8Pinned(
            # preserve_format: qdata stride ordering can encode a
            # transposed fp8 tensor, mirroring the NVFP4 handling.
            qdata=clone_to_pinned_cpu(f8.qdata),
            scale=clone_to_pinned_cpu(f8.scale),
            block_size=list(f8.block_size),
            mm_config=f8.mm_config,
            act_quant_kwargs=f8.act_quant_kwargs,
            kernel_preference=f8.kernel_preference,
            dtype=f8.dtype,
        )

    @staticmethod
    def cpu_param(
        state: _Float8Pinned, *, requires_grad: bool = False
    ) -> nn.Parameter:
        return nn.Parameter(
            _build_float8(state, state.qdata, state.scale),
            requires_grad=requires_grad,
        )

    @staticmethod
    def alloc_gpu(state: _Float8Pinned, device: torch.device) -> _Float8Gpu:
        return _Float8Gpu(
            qdata=empty_like_strided(state.qdata, device),
            scale=empty_like_strided(state.scale, device),
        )

    @staticmethod
    def gpu_param(
        pinned: _Float8Pinned,
        gpu_state: _Float8Gpu,
        *,
        requires_grad: bool = False,
    ) -> nn.Parameter:
        return nn.Parameter(
            _build_float8(pinned, gpu_state.qdata, gpu_state.scale),
            requires_grad=requires_grad,
        )

    @staticmethod
    def copy_to_gpu(
        src: _Float8Pinned, dst: _Float8Gpu, *, non_blocking: bool = False
    ) -> None:
        dst.qdata.copy_(src.qdata, non_blocking=non_blocking)
        dst.scale.copy_(src.scale, non_blocking=non_blocking)

    @staticmethod
    def copy_to_cpu(
        src: _Float8Gpu, dst: _Float8Pinned, *, non_blocking: bool = False
    ) -> None:
        # GPU storage is the identical fp8 bytes + scales, so D2H is a
        # lossless byte copy. Quant metadata lives on the pinned state
        # and is unaffected by GPU operations.
        dst.qdata.copy_(src.qdata, non_blocking=non_blocking)
        dst.scale.copy_(src.scale, non_blocking=non_blocking)

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        f8 = require_float8_tensor(t)
        return f8.dtype

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        return dequantize_float8_tensor(t)

    @staticmethod
    def requantize(t: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        return requantize_float8_tensor(t, like=like)

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        target_f8 = require_float8_tensor(target)
        src_f8 = require_float8_tensor(src)
        target_f8.qdata.copy_(src_f8.qdata)
        target_f8.scale.copy_(src_f8.scale)

    @staticmethod
    def cache_bytes(state: _Float8Pinned) -> int:
        return state.qdata.nbytes + state.scale.nbytes
