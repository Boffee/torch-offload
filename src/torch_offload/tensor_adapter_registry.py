"""Explicit tensor adapter selection and tensor identity.

The adapter classes themselves live in ``tensor_adapters.py`` and the
format-specific modules. This module is the one normal Python place that
knows the built-in dispatch order and adapter-defined tensor identity.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .bnb4bit_adapter import Bnb4bitAdapter
from .bnb8bit_adapter import Bnb8bitAdapter
from .dtensor_adapter import DTensorAdapter
from .float8_adapter import Float8Adapter
from .gguf_adapter import GgufAdapter
from .int4_tile_adapter import Int4TilePackedAdapter
from .int8_adapter import Int8Adapter
from .mx_adapter import MxAdapter
from .nvfp4_adapter import Nvfp4Adapter
from .quanto_adapter import QuantoAdapter
from .tensor_adapters import RegularAdapter, TensorAdapter

# Built-in adapters in dispatch order; first match wins. RegularAdapter is
# last — it matches only exact torch.Tensor/nn.Parameter, so structured
# subclasses reach their dedicated adapter first. DTensorAdapter is first: a
# DTensor is an outer wrapper (its local shard, possibly quantized, is moved
# by whatever adapter the registry selects for it), so it is checked before
# the local-shard adapters.
_BUILTIN_ADAPTERS: tuple[type[TensorAdapter[Any, Any]], ...] = (
    DTensorAdapter,
    QuantoAdapter,
    Bnb4bitAdapter,
    Bnb8bitAdapter,
    Nvfp4Adapter,
    MxAdapter,
    Float8Adapter,
    Int8Adapter,
    Int4TilePackedAdapter,
    GgufAdapter,
    RegularAdapter,
)


def select_adapter(t: torch.Tensor) -> TensorAdapter[Any, Any]:
    """Return the built-in adapter that handles ``t``."""
    for adapter_cls in _BUILTIN_ADAPTERS:
        if adapter_cls.matches(t):
            return adapter_cls()
    raise NotImplementedError(
        f"No TensorAdapter for tensor type {type(t).__name__!r}. "
        "Plain tensors are handled by RegularAdapter; tensor subclasses "
        "need a dedicated adapter."
    )


def tensor_id(t: torch.Tensor) -> tuple[Any, ...]:
    """Adapter-defined tensor identity for tied-weight detection."""
    return select_adapter(t).tensor_id(t)


def param_representation(param: torch.Tensor) -> torch.Tensor:
    """Return the tensor that carries ``param``'s adapter representation.

    For a plain :class:`nn.Parameter` (or bare tensor) this is ``param.data``
    — which for a quant tensor wrapped *inside* a Parameter (TorchAO
    ``Float8Tensor`` / ``NVFP4Tensor``, quanto ``WeightQBytesTensor``,
    ``GGUFWeight``) is the wrapped subclass itself, exactly what the adapter
    needs.

    A Parameter *subclass* that is itself the structured tensor — notably
    bitsandbytes ``Params4bit``, whose ``.data`` strips the quant state down
    to plain packed bytes — must be adapted as the object itself, or it would
    silently dispatch to :class:`RegularAdapter` and lose its quant state.
    """
    if type(param) is nn.Parameter or type(param) is torch.Tensor:
        return param.data
    return param


def param_tensor_id(param: nn.Parameter) -> tuple[Any, ...]:
    """Return an adapter-defined tensor identity for a parameter."""
    if param.numel() == 0:
        # Zero-sized tensors all share data_ptr()==0; key by object
        # identity so aliases of the same Parameter still dedupe.
        return ("__empty__", id(param))
    return tensor_id(param_representation(param))


def buffer_tensor_id(buffer: torch.Tensor) -> tuple[Any, ...]:
    """Return an adapter-defined tensor identity for a registered buffer."""
    if buffer.numel() == 0:
        return ("__empty_buf__", id(buffer))
    return tensor_id(buffer)


__all__ = [
    "buffer_tensor_id",
    "param_representation",
    "param_tensor_id",
    "select_adapter",
    "tensor_id",
]
