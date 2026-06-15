"""TorchAO ``Float8Tensor`` (scaled-fp8) adapter.

TorchAO's scaled-fp8 workflow stores weights as a tensor subclass with
fp8 bytes (``qdata``), per-row or per-tensor fp32 scales (``scale``),
and metadata controlling the matmul dispatch (``block_size``,
``mm_config``, ``kernel_preference``, ``act_quant_kwargs``). The shared
:class:`~torch_offload.torchao_structured_adapter.TorchaoStructuredAdapter`
base preserves that representation across pinned CPU and GPU storage;
this module supplies the Float8-specific hooks.

Beyond inference movement, this adapter opts into:

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
from typing import Any

import torch

from ._torchao_float8 import (
    create_float8_tensor,
    dequantize_float8_tensor,
    is_float8_tensor,
    requantize_float8_tensor,
    require_float8_tensor,
    validate_layout,
)
from .tensor_adapters import metadata_key
from .torchao_structured_adapter import (
    TorchaoGpu,
    TorchaoPinned,
    TorchaoStructuredAdapter,
    copy_storage,
)


@dataclass(slots=True, frozen=True)
class _Float8Meta:
    """Reconstruction metadata snapshot for a TorchAO scaled-fp8 tensor."""

    block_size: tuple[int, ...]
    mm_config: object | None
    act_quant_kwargs: object | None
    kernel_preference: object
    dtype: torch.dtype  # logical (pre-quantization) dtype


class Float8Adapter(TorchaoStructuredAdapter[_Float8Meta]):
    """Adapter for TorchAO ``Float8Tensor`` (scaled-fp8) weights."""

    _TAG = "torchao-float8"
    _STORAGE_NAMES = ("qdata", "scale")

    # --- per-format hooks -------------------------------------------------

    @staticmethod
    def _is_tensor(t: torch.Tensor) -> bool:
        return is_float8_tensor(t)

    @staticmethod
    def _validate_layout(t: torch.Tensor) -> None:
        validate_layout(t)

    @staticmethod
    def _require(t: torch.Tensor) -> Any:  # noqa: ANN401
        return require_float8_tensor(t)

    @staticmethod
    def _storage_of(t: Any) -> tuple[torch.Tensor | None, ...]:  # noqa: ANN401
        return (t.qdata, t.scale)

    @staticmethod
    def _meta_of(t: Any) -> _Float8Meta:  # noqa: ANN401
        return _Float8Meta(
            block_size=tuple(t.block_size),
            mm_config=t.mm_config,
            act_quant_kwargs=t.act_quant_kwargs,
            kernel_preference=t.kernel_preference,
            dtype=t.dtype,
        )

    @staticmethod
    def _reconstruct(
        storage: tuple[torch.Tensor | None, ...], meta: _Float8Meta
    ) -> torch.Tensor:
        qdata, scale = storage
        assert qdata is not None
        assert scale is not None
        return create_float8_tensor(
            qdata,
            scale,
            list(meta.block_size),
            meta.mm_config,
            meta.act_quant_kwargs,
            meta.kernel_preference,
            meta.dtype,
        )

    @staticmethod
    def _id_metadata(t: Any) -> tuple[object, ...]:  # noqa: ANN401
        return (
            tuple(t.block_size),
            t.dtype,
            metadata_key(t.mm_config),
            metadata_key(t.kernel_preference),
            metadata_key(t.act_quant_kwargs),
        )

    @classmethod
    def _layout_metadata(cls, t: Any) -> tuple[object, ...]:  # noqa: ANN401
        # Float8 diverges from identity metadata: drop the logical dtype,
        # which layout_signature's standard dtype slot already carries.
        return (
            tuple(t.block_size),
            metadata_key(t.mm_config),
            metadata_key(t.kernel_preference),
            metadata_key(t.act_quant_kwargs),
        )

    @staticmethod
    def _compute_dtype(t: Any) -> torch.dtype:  # noqa: ANN401
        return t.dtype

    # --- capabilities beyond inference movement ---------------------------

    @staticmethod
    def copy_to_cpu(
        src: TorchaoGpu, dst: TorchaoPinned[_Float8Meta], *, non_blocking: bool = False
    ) -> None:
        # GPU storage is the identical fp8 bytes + scales, so D2H is a
        # lossless byte copy. Quant metadata lives on the pinned state.
        copy_storage(src.storage, dst.storage, non_blocking=non_blocking)

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
