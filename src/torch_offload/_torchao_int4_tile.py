"""Internal optional-import module for TorchAO INT4 tile-packed support.

Single source of truth for the TorchAO ``Int4TilePackedTo4dTensor``
layout this repo needs to move int4 weight-only weights through
:class:`PinnedParam`. TorchAO's public workflow creates these weights via
``quantize_(..., Int4WeightOnlyConfig(int4_packing_format=
Int4PackingFormat.TILE_PACKED_TO_4D))``; the adapter only preserves and
moves those already-quantized tensors.

``Int4TilePackedTo4dTensor`` is the CUDA-native (tinygemm) int4 variant —
no external kernel library required. It carries one required packed int32
storage tensor (``qdata``, a 4-D tile-packed layout whose shape differs
from the logical weight shape), a combined ``scale_and_zero`` tensor, an
optional ``act_pre_scale``, and metadata ``block_size`` plus the logical
``shape`` (needed to reconstruct the wrapper, since ``qdata`` is packed).
"""

from __future__ import annotations

from typing import Any

import torch

LAYOUT_ATTRS = (
    "qdata",
    "scale_and_zero",
    "act_pre_scale",
    "block_size",
    "shape",
)
"""Attributes this repo reads from a TorchAO ``Int4TilePackedTo4dTensor``."""


try:
    from torchao.quantization.quantize_.workflows import (
        Int4TilePackedTo4dTensor,
    )

    TORCHAO_INT4_TILE_AVAILABLE = True
except ImportError:
    TORCHAO_INT4_TILE_AVAILABLE = False
    Int4TilePackedTo4dTensor: Any = None


def is_int4_tile_tensor(t: object) -> bool:
    """Return whether ``t`` is a TorchAO ``Int4TilePackedTo4dTensor``."""
    return TORCHAO_INT4_TILE_AVAILABLE and isinstance(
        t, Int4TilePackedTo4dTensor
    )


def require_int4_tile_tensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated tile-packed int4 tensor, or raise."""
    if not is_int4_tile_tensor(t):
        raise TypeError(
            f"expected TorchAO Int4TilePackedTo4dTensor, "
            f"got {type(t).__name__}"
        )
    validate_layout(t)
    return t


def create_int4_tile_tensor(
    qdata: torch.Tensor,
    scale_and_zero: torch.Tensor,
    block_size: list[int],
    shape: torch.Size,
    act_pre_scale: torch.Tensor | None,
) -> torch.Tensor:
    """Rebuild a TorchAO ``Int4TilePackedTo4dTensor`` from raw storage +
    metadata. Re-wrapping already-packed bytes is a pure constructor call
    (no int4-pack kernel), so it works on CPU as well as CUDA."""
    if not TORCHAO_INT4_TILE_AVAILABLE:
        raise RuntimeError(
            "torchao is required to create an Int4TilePackedTo4dTensor"
        )
    return Int4TilePackedTo4dTensor(
        qdata,
        scale_and_zero,
        block_size,
        shape,
        act_pre_scale=act_pre_scale,
    )


def validate_layout(t: torch.Tensor) -> None:
    """Raise if ``t`` is missing the tile-packed int4 attributes we preserve."""
    missing = [a for a in LAYOUT_ATTRS if not hasattr(t, a)]
    if not missing:
        return
    raise RuntimeError(
        f"Int4TilePackedTo4dTensor is missing expected attributes "
        f"{missing!r}; this repo is pinned to a layout that exposes "
        f"{LAYOUT_ATTRS}. TorchAO likely refactored the wrapper class — "
        "upgrade torch-offload to match."
    )
