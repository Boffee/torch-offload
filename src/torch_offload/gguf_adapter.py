"""GGUF tensor adapter for the memory subsystem.

Registers :class:`GgufAdapter` when the ``gguf`` package is installed,
enabling :class:`PinnedParamBuffer` to handle GGUF-quantized weights.

Weights are stored in their compact quantized form in pinned CPU
memory.  On GPU transfer the raw bytes are DMA'd to a GPU staging
buffer and dequantized on-device to bfloat16 using pure-PyTorch
bit-manipulation ops (vendored from HuggingFace diffusers / city96).

Memory layout::

    Pinned CPU:  compact GGUF bytes  (e.g. Q4_K → ~4.5 bits/weight)
         ↓  DMA (small transfer)
    GPU staging: compact GGUF bytes  (uint8)
         ↓  on-device dequant
    GPU output:  bf16 weight tensor  (full precision for matmul)

The staging + output buffers are pre-allocated once per pool slot and
reused across block loads, matching the existing pooled streaming path.

Auto-registers when this module is imported.  Import fails silently if
the ``gguf`` package is not installed — GGUF support is optional.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .tensor_adapters import register_adapter

try:
    from .gguf_dequant import dequant_shape, dequantize

    _GGUF_AVAILABLE = True
except ImportError:
    _GGUF_AVAILABLE = False

__all__ = ["GGUFWeight", "GgufAdapter"]


# -------------------------------------------------------------------
# Tensor subclass — carries quant_type through PyTorch operations
# -------------------------------------------------------------------


class GGUFWeight(torch.Tensor):
    """``torch.Tensor`` subclass wrapping GGUF-quantized raw bytes.

    The underlying data is ``uint8``; the ``quant_type`` attribute
    (a ``gguf.GGMLQuantizationType`` int) tells the adapter which
    dequant function to use.

    PyTorch preserves the subclass through ``.data``, ``.clone()``,
    ``.detach()`` etc., but drops arbitrary instance attributes.
    ``__torch_function__`` re-attaches ``quant_type`` to any result
    that is still a ``GGUFWeight`` instance.
    """

    quant_type: int

    @staticmethod
    def __new__(cls, data: torch.Tensor, *, quant_type: int) -> GGUFWeight:  # noqa: ANN001
        t = torch.Tensor._make_subclass(cls, data, False)
        t.quant_type = quant_type
        return t

    @classmethod
    def __torch_function__(
        cls, func, types, args=(), kwargs=None,  # noqa: ANN001
    ) -> object:
        result = super().__torch_function__(func, types, args, kwargs or {})
        qt = _extract_quant_type(args)
        if qt is not None:
            if isinstance(result, cls):
                result.quant_type = qt
            elif isinstance(result, (list, tuple)):
                for r in result:
                    if isinstance(r, cls):
                        r.quant_type = qt
        return result

    def __repr__(self) -> str:
        import gguf as _gguf  # noqa: PLC0415

        try:
            name = _gguf.GGMLQuantizationType(self.quant_type).name
        except (ValueError, AttributeError):
            name = str(self.quant_type)
        return (
            f"GGUFWeight(quant_type={name}, packed_shape={list(self.shape)}, "
            f"device={self.device})"
        )


def _extract_quant_type(args: tuple) -> int | None:
    for a in args:
        if isinstance(a, GGUFWeight):
            return a.quant_type
        if isinstance(a, (list, tuple)):
            for item in a:
                if isinstance(item, GGUFWeight):
                    return item.quant_type
    return None


# -------------------------------------------------------------------
# Pinned / GPU state dataclasses
# -------------------------------------------------------------------


@dataclass(slots=True)
class _GgufPinned:
    """Pinned-CPU state: compact quantized bytes + metadata."""

    data: torch.Tensor         # pinned uint8 raw GGUF bytes
    quant_type: int             # GGMLQuantizationType value
    dequant_shape: torch.Size   # logical weight shape after dequant
    compute_dtype: torch.dtype  # target dtype (default bf16)


@dataclass(slots=True)
class _GgufGpu:
    """GPU state: staging buffer for DMA + output buffer for dequant."""

    staging: torch.Tensor   # uint8, same shape as pinned data
    dequant: torch.Tensor   # compute_dtype, logical weight shape


# -------------------------------------------------------------------
# Adapter
# -------------------------------------------------------------------


class GgufAdapter:
    """TensorAdapter for GGUF-quantized weights.

    Keeps weights in compact quantized form in pinned CPU memory.
    On :meth:`copy_to_gpu`:

    1. DMA the small quantized bytes to a GPU staging buffer.
    2. Dequantize on-device into a pre-allocated bf16 output buffer.

    The ``gpu_param`` wraps the output buffer, so the model sees
    standard bf16 weights during the forward pass.
    """

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        return isinstance(t, GGUFWeight) and t.dtype == torch.uint8

    @staticmethod
    def storage_key(t: torch.Tensor) -> tuple:
        raw = t.as_subclass(torch.Tensor)
        return (
            "gguf",
            raw.untyped_storage().data_ptr(),
            raw.storage_offset(),
            tuple(raw.shape),
            raw.stride(),
            t.quant_type,  # type: ignore[union-attr]
        )

    @staticmethod
    def clone_pin(t: torch.Tensor) -> _GgufPinned:
        raw = t.as_subclass(torch.Tensor)
        if raw.is_cuda:
            pinned = torch.empty_like(raw, device="cpu").pin_memory()
            pinned.copy_(raw)
        else:
            pinned = raw.contiguous().clone().pin_memory()
        qt: int = t.quant_type  # type: ignore[union-attr]
        shape = torch.Size(dequant_shape(tuple(raw.shape), qt))
        return _GgufPinned(
            data=pinned,
            quant_type=qt,
            dequant_shape=shape,
            compute_dtype=torch.bfloat16,
        )

    @staticmethod
    def cpu_param(state: _GgufPinned) -> nn.Parameter:
        return nn.Parameter(
            GGUFWeight(state.data, quant_type=state.quant_type),
            requires_grad=False,
        )

    @staticmethod
    def alloc_gpu(state: _GgufPinned, device: torch.device) -> _GgufGpu:
        staging = torch.empty(
            state.data.shape, dtype=torch.uint8, device=device,
        )
        dequant = torch.empty(
            state.dequant_shape, dtype=state.compute_dtype, device=device,
        )
        return _GgufGpu(staging=staging, dequant=dequant)

    @staticmethod
    def gpu_param(pinned: _GgufPinned, gpu_state: _GgufGpu) -> nn.Parameter:  # noqa: ARG004
        return nn.Parameter(gpu_state.dequant, requires_grad=False)

    @staticmethod
    def copy_to_gpu(
        src: _GgufPinned, dst: _GgufGpu, *, non_blocking: bool = False,
    ) -> None:
        # Stage 1: DMA compact bytes to GPU (small transfer).
        dst.staging.copy_(src.data, non_blocking=non_blocking)
        # Stage 2: dequantize on-device into pre-allocated output.
        # dequantize() returns a new tensor; copy into the persistent
        # output buffer so gpu_param().data sees the update.
        result = dequantize(dst.staging, src.quant_type, dtype=src.compute_dtype)
        dst.dequant.copy_(result)

    @staticmethod
    def cache_bytes(state: _GgufPinned) -> int:
        return state.data.nbytes


if _GGUF_AVAILABLE:
    register_adapter(GgufAdapter)
