"""Triton kernels for static per-tensor FP8 LoRA merges."""

# Triton JIT kernel signatures intentionally use untyped pointer parameters
# and upper-case constexpr names.
# ruff: noqa: ANN001, ANN202, N803, PLR0913
# pyright: reportCallIssue=false

from __future__ import annotations

import torch
import triton
import triton.language as tl

_COMPUTE_FP16 = 0
_COMPUTE_BF16 = 1
_COMPUTE_FP32 = 2
_REDUCTION_BLOCK = 8192


@triton.jit
def _merge_dense_kernel(
    qdata_ptr,
    scale_ptr,
    b_ptr,
    a_ptr,
    dense_ptr,
    tile_max_ptr,
    strength,
    M,
    N,
    K: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        b = tl.load(
            b_ptr + offsets_m[:, None] * K + offsets_k[None, :],
            mask=(offsets_m[:, None] < M) & (offsets_k[None, :] < K),
            other=0.0,
        )
        a = tl.load(
            a_ptr + offsets_k[:, None] * N + offsets_n[None, :],
            mask=(offsets_k[:, None] < K) & (offsets_n[None, :] < N),
            other=0.0,
        )
        if COMPUTE_DTYPE == 2:
            accumulator += tl.dot(b, a, input_precision="ieee")
        else:
            accumulator += tl.dot(b, a)

    offsets = offsets_m[:, None] * N + offsets_n[None, :]
    mask = (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    weight_scale = tl.load(scale_ptr)
    base = tl.load(qdata_ptr + offsets, mask=mask, other=0.0)
    base = base.to(tl.float32) * weight_scale
    if COMPUTE_DTYPE == 0:
        base = base.to(tl.float16)
    elif COMPUTE_DTYPE == 1:
        base = base.to(tl.bfloat16)

    merged = base.to(tl.float32) + accumulator * strength
    if COMPUTE_DTYPE == 0:
        merged = merged.to(tl.float16)
    elif COMPUTE_DTYPE == 1:
        merged = merged.to(tl.bfloat16)
    tl.store(dense_ptr + offsets, merged, mask=mask)

    absolute = tl.where(mask, tl.abs(merged.to(tl.float32)), 0.0)
    tile_max = tl.max(tl.max(absolute, axis=1), axis=0)
    tl.store(tile_max_ptr + pid, tile_max)


@triton.jit
def _reduce_max_kernel(
    input_ptr,
    output_ptr,
    NUM_VALUES,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        input_ptr + offsets,
        mask=offsets < NUM_VALUES,
        other=0.0,
    )
    tl.store(output_ptr + tl.program_id(axis=0), tl.max(values, axis=0))


@triton.jit
def _reduce_scale_kernel(
    tile_max_ptr,
    output_scale_ptr,
    NUM_TILES,
    FP8_LIMIT: tl.constexpr,
    COMPUTE_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(
        tile_max_ptr + offsets,
        mask=offsets < NUM_TILES,
        other=0.0,
    )
    max_abs = tl.max(values, axis=0)
    scale = max_abs / FP8_LIMIT
    if COMPUTE_DTYPE == 0:
        scale = scale.to(tl.float16).to(tl.float32)
    elif COMPUTE_DTYPE == 1:
        scale = scale.to(tl.bfloat16).to(tl.float32)
    scale = tl.where(scale == 0.0, 1.1920928955078125e-07, scale)
    tl.store(output_scale_ptr, scale)


@triton.jit
def _quantize_kernel(
    dense_ptr,
    scale_ptr,
    output_ptr,
    NUMEL,
    FP8_LIMIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL
    dense = tl.load(dense_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr)
    scaled = dense / scale
    scaled = tl.minimum(
        tl.maximum(scaled, -FP8_LIMIT),
        FP8_LIMIT,
    )
    tl.store(output_ptr + offsets, scaled, mask=mask)


def _compute_dtype_id(dtype: torch.dtype) -> int:
    if dtype is torch.float16:
        return _COMPUTE_FP16
    if dtype is torch.bfloat16:
        return _COMPUTE_BF16
    if dtype is torch.float32:
        return _COMPUTE_FP32
    raise ValueError(
        "Triton static-FP8 merge supports float16, bfloat16, and float32 "
        f"LoRA factors, got {dtype}."
    )


def merge_static_float8_lora(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw FP8 buffers after one static-per-tensor LoRA merge."""
    if qdata.device.type != "cuda":
        raise ValueError("Triton static-FP8 merge requires CUDA tensors.")
    if qdata.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(
            "Triton static-FP8 merge supports E4M3FN and E5M2 storage, "
            f"got {qdata.dtype}."
        )
    if b.dtype is not a.dtype:
        raise ValueError(
            "Triton static-FP8 merge requires matching LoRA factor dtypes."
        )
    compute_dtype = _compute_dtype_id(b.dtype)
    if qdata.ndim != 2 or b.ndim != 2 or a.ndim != 2:
        raise ValueError("Triton static-FP8 merge expects rank-two tensors.")
    if (
        qdata.device != scale.device
        or qdata.device != b.device
        or qdata.device != a.device
    ):
        raise ValueError(
            "Triton static-FP8 merge requires all tensors on one CUDA device."
        )
    if scale.dtype is not torch.float32 or scale.numel() != 1:
        raise ValueError("Triton static-FP8 merge expects one float32 per-tensor scale.")

    rows, cols = qdata.shape
    rank = a.shape[0]
    if rows == 0 or cols == 0 or rank == 0:
        raise ValueError(
            "Triton static-FP8 merge requires non-empty weight and factors."
        )
    if b.shape != (rows, rank) or a.shape[1] != cols:
        raise ValueError("LoRA factors do not match the FP8 weight shape.")

    qdata = qdata.contiguous()
    b = b.contiguous()
    a = a.contiguous()

    block_m = 64
    block_n = 128
    block_k = 16 if rank <= 16 else 32
    num_tiles = ((rows + block_m - 1) // block_m) * (
        (cols + block_n - 1) // block_n
    )
    dense = torch.empty_like(qdata, dtype=b.dtype)
    tile_max = torch.empty(num_tiles, device=qdata.device, dtype=torch.float32)
    output_qdata = torch.empty_like(qdata)
    output_scale = torch.empty_like(scale)

    _merge_dense_kernel[(num_tiles,)](
        qdata,
        scale,
        b,
        a,
        dense,
        tile_max,
        strength,
        M=rows,
        N=cols,
        K=rank,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=8,
    )

    reduction_inputs = [tile_max]
    num_values = num_tiles
    while num_values > _REDUCTION_BLOCK:
        output_count = (num_values + _REDUCTION_BLOCK - 1) // _REDUCTION_BLOCK
        reduced = torch.empty(
            output_count,
            device=qdata.device,
            dtype=torch.float32,
        )
        _reduce_max_kernel[(output_count,)](
            reduction_inputs[-1],
            reduced,
            NUM_VALUES=num_values,
            BLOCK_SIZE=_REDUCTION_BLOCK,
            num_warps=8,
        )
        reduction_inputs.append(reduced)
        num_values = output_count

    fp8_limit = torch.finfo(qdata.dtype).max
    _reduce_scale_kernel[(1,)](
        reduction_inputs[-1],
        output_scale,
        NUM_TILES=num_values,
        FP8_LIMIT=fp8_limit,
        COMPUTE_DTYPE=compute_dtype,
        BLOCK_SIZE=_REDUCTION_BLOCK,
        num_warps=8,
    )

    quant_block = 1024
    quant_grid = (qdata.numel() + quant_block - 1) // quant_block
    _quantize_kernel[(quant_grid,)](
        dense,
        output_scale,
        output_qdata,
        NUMEL=qdata.numel(),
        FP8_LIMIT=fp8_limit,
        BLOCK_SIZE=quant_block,
        num_warps=8,
    )
    return output_qdata, output_scale
