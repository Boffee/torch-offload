"""bitsandbytes 8-bit (``Int8Params`` / LLM.int8) adapter.

bitsandbytes 8-bit weights are ``Int8Params`` tensors (an ``nn.Parameter``
subclass) carrying two pieces: ``CB`` (the row-major int8 weight, also
``weight.data``) and ``SCB`` (the per-output-row float32 scale). Like
:class:`~torch_offload.bnb4bit_adapter.Bnb4bitAdapter` this is a
decompose-on-pin / reconstruct-on-move, inference-only adapter: each
activate builds a fresh ``Int8Params`` installed via registry replacement,
so optimizers keyed on the pre-wrap Parameter are orphaned.

Two things differ from 4-bit, both because int8 keeps no self-describing
``QuantState``:

- The representation is just two tensors (``CB`` + ``SCB``) — no packed
  metadata blob, no codebook, no nested double-quant. Movement is simpler.
- ``Int8Params`` carries ``CB``/``SCB`` only **before** the owning
  ``Linear8bitLt`` runs its first forward. The first forward migrates them
  onto the module's ``MatmulLtState`` and nulls them on the weight. A
  tensor-scoped adapter cannot see the module, so it must pin a pre-forward
  weight; :meth:`Bnb8bitAdapter.matches` (via ``validate_int8_layout``)
  raises on a post-forward weight rather than silently capturing ``None``.

This is safe across activations because the reconstructed ``Int8Params``
carries ``CB``, so each forward re-runs ``init_8bit_state`` and repopulates
the module state from the freshly-installed weight — the offloader never
has to touch the owning module.

Reaches into bitsandbytes' int8 layout through :mod:`._bnb`; if
bitsandbytes refactors, :meth:`Bnb8bitAdapter.matches` fails with a clear
validation error. Selected by :mod:`tensor_adapter_registry`. Importing
fails silently if bitsandbytes is not installed — 8-bit support is optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ._bnb import (
    build_int8_params,
    dequantize_int8_params,
    is_int8_params,
    requantize_int8_params,
    require_int8_params,
    validate_int8_layout,
)
from .tensor_adapters import (
    clone_to_pinned_cpu,
    empty_like_strided,
    optional_tensor_id,
    tensor_layout,
)


@dataclass(slots=True)
class _Bnb8bitPinned:
    """Pinned-CPU state for an 8-bit tensor: the int8 weight (``CB``) and
    the per-row float32 scale (``SCB``)."""

    data: torch.Tensor  # pinned int8 CB (== weight.data)
    scb: torch.Tensor   # pinned float32 per-row scale


@dataclass(slots=True)
class _Bnb8bitGpu:
    """GPU state for an 8-bit tensor: the int8 weight + the per-row scale."""

    data: torch.Tensor
    scb: torch.Tensor


class Bnb8bitAdapter:
    """Adapter for ``bitsandbytes.nn.Int8Params`` (LLM.int8).

    Decompose-on-pin (``CB`` + ``SCB``), reconstruct a fresh ``Int8Params``
    on each activate. Inference-only. Must pin a pre-forward weight — int8
    quant state migrates onto the owning module after the first forward,
    out of a tensor-scoped adapter's reach.
    """

    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        if not is_int8_params(t):
            return False
        # Validate the layout we read in clone_pin / reconstruction, and
        # that the weight is pre-forward (CB/SCB still present). Cheap; runs
        # on every dispatch.
        validate_int8_layout(t)
        return True

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple:
        # Composite identity: the int8 weight + the per-row scale. Two
        # Int8Params sharing the same CB/SCB storage are the same logical
        # tensor and dedup safely.
        qt = require_int8_params(t)
        # CB is the logical weight tensor; optional_tensor_id already embeds
        # its shape, so no separate shape field is needed (cf. 4-bit, where
        # the packed qt.data shape differs from the logical quant_state.shape).
        return (
            "bnb8bit",
            optional_tensor_id(qt.CB),
            optional_tensor_id(qt.SCB),
        )

    @staticmethod
    def layout_signature(t: torch.Tensor) -> tuple:
        qt = require_int8_params(t)
        return (
            tuple(qt.CB.shape),
            ("CB", tensor_layout(qt.CB)),
            ("SCB", tensor_layout(qt.SCB)),
        )

    @staticmethod
    def clone_pin(t: torch.Tensor) -> _Bnb8bitPinned:
        qt = require_int8_params(t)
        return _Bnb8bitPinned(
            data=clone_to_pinned_cpu(
                qt.CB, memory_format=torch.contiguous_format
            ),
            scb=clone_to_pinned_cpu(
                qt.SCB, memory_format=torch.contiguous_format
            ),
        )

    @staticmethod
    def cpu_param(
        state: _Bnb8bitPinned, *, requires_grad: bool = False
    ) -> nn.Parameter:
        # Int8Params is itself an nn.Parameter subclass.
        return build_int8_params(
            state.data, state.scb, requires_grad=requires_grad
        )

    @staticmethod
    def alloc_gpu(state: _Bnb8bitPinned, device: torch.device) -> _Bnb8bitGpu:
        return _Bnb8bitGpu(
            data=empty_like_strided(state.data, device),
            scb=empty_like_strided(state.scb, device),
        )

    @staticmethod
    def gpu_param(
        pinned: _Bnb8bitPinned,
        gpu_state: _Bnb8bitGpu,
        *,
        requires_grad: bool = False,
    ) -> nn.Parameter:
        _ = pinned
        # int8 carries no host-resident metadata (unlike the 4-bit blob), so
        # the wrapper is built purely from the GPU buffers. Reconstruction
        # aliases them, so the later copy_to_gpu DMA is visible here, and the
        # rebuilt wrapper carries CB so the next forward re-inits module state.
        return build_int8_params(
            gpu_state.data, gpu_state.scb, requires_grad=requires_grad
        )

    @staticmethod
    def copy_to_gpu(
        src: _Bnb8bitPinned, dst: _Bnb8bitGpu, *, non_blocking: bool = False
    ) -> None:
        dst.data.copy_(src.data, non_blocking=non_blocking)
        dst.scb.copy_(src.scb, non_blocking=non_blocking)

    @staticmethod
    def compute_dtype(t: torch.Tensor) -> torch.dtype:
        require_int8_params(t)
        # LLM.int8 dequantizes/accumulates in fp16; there is no module-level
        # compute_dtype the routed-LoRA path can read off Linear8bitLt.
        return torch.float16

    @staticmethod
    def dequantize(t: torch.Tensor) -> torch.Tensor:
        return dequantize_int8_params(t)

    @staticmethod
    def requantize(t: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
        return requantize_int8_params(t, like=like)

    @staticmethod
    def rearm_after_load(param: nn.Parameter, gpu_state: _Bnb8bitGpu) -> None:
        # The previous forward migrated CB/SCB onto the owning module and
        # nulled them here; re-point them at the freshly-loaded buffers so the
        # next forward re-runs init_8bit_state from the current data. Pool
        # buffers are shared across streamed blocks, so without this the
        # migrated module state would read another block's bytes. CB is None
        # at this point (post-forward), so this cannot go through the
        # CB-present validation in require_int8_params.
        wrapper: Any = param
        wrapper.CB = gpu_state.data
        wrapper.SCB = gpu_state.scb

    @staticmethod
    def copy_into(src: torch.Tensor, *, target: torch.Tensor) -> None:
        # Preserve target identity/storage: overwrite CB and SCB in place.
        # CB and weight.data alias the same storage, so copying CB updates
        # both.
        target_qt = require_int8_params(target)
        src_qt = require_int8_params(src)
        target_qt.CB.copy_(src_qt.CB)
        target_qt.SCB.copy_(src_qt.SCB)

    @staticmethod
    def cache_bytes(state: _Bnb8bitPinned) -> int:
        return (
            state.data.numel() * state.data.element_size()
            + state.scb.numel() * state.scb.element_size()
        )
