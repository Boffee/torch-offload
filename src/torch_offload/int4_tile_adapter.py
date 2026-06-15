"""TorchAO ``Int4TilePackedTo4dTensor`` adapter (CUDA-native int4 WO).

TorchAO's tile-packed int4 weight-only workflow stores weights as a
tensor subclass with a 4-D tile-packed int32 ``qdata`` (whose shape
differs from the logical weight shape), a combined ``scale_and_zero``
tensor, an optional ``act_pre_scale``, and metadata ``block_size`` plus
the logical ``shape``. The shared
:class:`~torch_offload.torchao_structured_adapter.TorchaoStructuredAdapter`
base preserves that representation across pinned CPU and GPU storage;
this module supplies the tile-packed-specific hooks.

This is the CUDA-native (tinygemm) int4 variant — it needs no external
kernel library. The matmul runs on CUDA; re-wrapping already-packed bytes
is a pure constructor call, so the pinned-CPU representation round-trips
cleanly.

The adapter exposes inference movement only. INT4 model weights are
treated as frozen: no CPU round-trip, no trainable ``Parameter.data``
swap, and no activation-scoped dense ``addmm_`` LoRA merge. Routed LoRA
remains possible when the owning module is a logical ``nn.Linear`` with
compatible shape/dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ._torchao_int4_tile import (
    create_int4_tile_tensor,
    is_int4_tile_tensor,
    require_int4_tile_tensor,
    validate_layout,
)
from .torchao_structured_adapter import TorchaoStructuredAdapter


@dataclass(slots=True, frozen=True)
class _Int4TileMeta:
    """Reconstruction metadata snapshot for a tile-packed int4 tensor."""

    block_size: tuple[int, ...]
    shape: tuple[int, ...]  # logical weight shape (qdata is packed)


class Int4TilePackedAdapter(TorchaoStructuredAdapter[_Int4TileMeta]):
    """Adapter for TorchAO ``Int4TilePackedTo4dTensor`` weights."""

    _TAG = "torchao-int4-tile-packed"
    _STORAGE_NAMES = ("qdata", "scale_and_zero", "act_pre_scale")

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return is_int4_tile_tensor(t)

    @staticmethod
    def _validate_layout(t: torch.Tensor) -> None:
        validate_layout(t)

    @staticmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_int4_tile_tensor(t)

    @staticmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        return (t.qdata, t.scale_and_zero, t.act_pre_scale)

    @staticmethod
    def _meta_of(t: Any) -> _Int4TileMeta:  # noqa: ANN401
        return _Int4TileMeta(
            block_size=tuple(t.block_size),
            shape=tuple(t.shape),
        )

    @staticmethod
    def _reconstruct(
        storage: tuple[torch.Tensor | None, ...], meta: _Int4TileMeta
    ) -> torch.Tensor:
        qdata, scale_and_zero, act_pre_scale = storage
        assert qdata is not None
        assert scale_and_zero is not None
        return create_int4_tile_tensor(
            qdata,
            scale_and_zero,
            list(meta.block_size),
            torch.Size(meta.shape),
            act_pre_scale,
        )

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        # Logical shape is already carried by the tensor_id skeleton
        # (tuple(t.shape)); block_size (group size) is the distinguishing
        # extra. The wrapper has no act_quant_kwargs.
        return (tuple(t.block_size),)

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        # Logical compute dtype; derived from scale_and_zero, surfaced as
        # the wrapper's .dtype (e.g. bf16).
        return t.dtype
