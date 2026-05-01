"""GPU-side dequantization for GGUF quantized tensors.

Pure PyTorch tensor operations — no C extensions required.  Vendored
from HuggingFace diffusers (Apache-2.0) which itself ported the ops
from city96/ComfyUI-GGUF.

Requires the ``gguf`` package (``pip install gguf``) for quant-type
constants; all computation is native PyTorch and runs on any device.

Vendored code — lint rules relaxed intentionally.
"""
# ruff: noqa: N802, ARG001, E501, RUF005

from __future__ import annotations

import gguf
import torch

GGML_QUANT_SIZES: dict[int, tuple[int, int]] = gguf.GGML_QUANT_SIZES

QK_K = 256
K_SCALE_SIZE = 12


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _to_uint32(x: torch.Tensor) -> torch.Tensor:
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def _split_block_dims(blocks: torch.Tensor, *args: int) -> tuple[torch.Tensor, ...]:
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def _get_scale_min(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8).reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape((n_blocks, 8)), mn.reshape((n_blocks, 8))


# -------------------------------------------------------------------
# Per-type dequant functions
# -------------------------------------------------------------------

def _dequantize_Q8_0(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    d, x = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def _dequantize_Q5_1(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, m, qh, qs = _split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs_out = ql | (qh << 4)
    return (d * qs_out) + m


def _dequantize_Q5_0(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, qh, qs = _split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs_out = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs_out


def _dequantize_Q4_1(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, m, qs = _split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    return (d * qs) + m


def _dequantize_Q4_0(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs


def _dequantize_Q6_K(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = _split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))


def _dequantize_Q5_K(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.arange(
        0, 8, device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)
    return (d * q - dm).reshape((n_blocks, QK_K))


def _dequantize_Q4_K(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))


def _dequantize_Q3_K(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = _split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales_out = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales_out = scales_out.to(torch.int8) - 32
    dl = (d * scales_out).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.arange(
        0, 8, device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)
    return (dl * q).reshape((n_blocks, QK_K))


def _dequantize_Q2_K(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = _split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))


def _dequantize_BF16(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


def _dequantize_F16(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    return blocks.view(torch.float16)


def _dequantize_F32(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    return blocks.view(torch.float32)


def _dequantize_IQ4_NL(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    kvalues = torch.tensor(
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
        dtype=torch.float32, device=blocks.device,
    )
    n_blocks = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 15).reshape((n_blocks, -1)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], 16), 2, qs)
    qs = qs.squeeze(-1).to(dtype)
    return d * qs


def _dequantize_IQ4_XS(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    kvalues = torch.tensor(
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
        dtype=torch.float32, device=blocks.device,
    )
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = _split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = scales_h.view(torch.int16)
    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, 1, -1)) >> torch.tensor(
        [2 * i for i in range(QK_K // 32)], device=blocks.device, dtype=torch.uint8
    ).reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)) & 0x03
    scales_out = (scales_l | (scales_h << 4)) - 32
    dl = (d * scales_out.to(dtype)).reshape((n_blocks, -1, 1))
    shifts_q = torch.tensor([0, 4], device=blocks.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shifts_q
    qs = (qs & 15).reshape((n_blocks, -1, 32)).to(torch.int64)
    kvalues = kvalues.view(1, 1, 1, 16)
    qs = qs.unsqueeze(-1)
    qs = torch.gather(kvalues.expand(qs.shape[0], qs.shape[1], qs.shape[2], 16), 3, qs)
    qs = qs.squeeze(-1).to(dtype)
    return (dl * qs).reshape(n_blocks, -1)


# -------------------------------------------------------------------
# Dispatch table and public API
# -------------------------------------------------------------------

_DEQUANT_FUNCTIONS = {
    gguf.GGMLQuantizationType.F32: _dequantize_F32,
    gguf.GGMLQuantizationType.F16: _dequantize_F16,
    gguf.GGMLQuantizationType.BF16: _dequantize_BF16,
    gguf.GGMLQuantizationType.Q8_0: _dequantize_Q8_0,
    gguf.GGMLQuantizationType.Q5_1: _dequantize_Q5_1,
    gguf.GGMLQuantizationType.Q5_0: _dequantize_Q5_0,
    gguf.GGMLQuantizationType.Q4_1: _dequantize_Q4_1,
    gguf.GGMLQuantizationType.Q4_0: _dequantize_Q4_0,
    gguf.GGMLQuantizationType.Q6_K: _dequantize_Q6_K,
    gguf.GGMLQuantizationType.Q5_K: _dequantize_Q5_K,
    gguf.GGMLQuantizationType.Q4_K: _dequantize_Q4_K,
    gguf.GGMLQuantizationType.Q3_K: _dequantize_Q3_K,
    gguf.GGMLQuantizationType.Q2_K: _dequantize_Q2_K,
    gguf.GGMLQuantizationType.IQ4_NL: _dequantize_IQ4_NL,
    gguf.GGMLQuantizationType.IQ4_XS: _dequantize_IQ4_XS,
}


def dequant_shape(packed_shape: tuple[int, ...], quant_type: int) -> tuple[int, ...]:
    """Compute logical tensor shape from the packed byte-level shape."""
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    return (*packed_shape[:-1], packed_shape[-1] // type_size * block_size)


def dequantize(
    data: torch.Tensor,
    quant_type: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize GGUF raw bytes to a float tensor.

    Works on any device (CPU or CUDA).  For GPU-resident ``data`` the
    bit-manipulation runs entirely as PyTorch CUDA kernels.
    """
    fn = _DEQUANT_FUNCTIONS.get(quant_type)
    if fn is None:
        raise NotImplementedError(
            f"Unsupported GGUF quant type: {gguf.GGMLQuantizationType(quant_type)!r}"
        )

    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    shape = dequant_shape(tuple(data.shape), quant_type)

    raw = data.view(torch.uint8)
    n_blocks = raw.numel() // type_size
    blocks = raw.reshape((n_blocks, type_size))

    result = fn(blocks, block_size, type_size, dtype=dtype)
    result = result.reshape(shape)
    if result.dtype != dtype:
        result = result.to(dtype)
    return result
