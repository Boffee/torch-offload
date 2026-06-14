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
from .float8_adapter import Float8Adapter
from .gguf_adapter import GgufAdapter
from .nvfp4_adapter import Nvfp4Adapter
from .quanto_adapter import QuantoAdapter
from .tensor_adapters import RegularAdapter, TensorAdapter


def select_adapter(t: torch.Tensor) -> TensorAdapter[Any, Any]:
    """Return the built-in adapter that handles ``t``."""
    if QuantoAdapter.matches(t):
        return QuantoAdapter()
    if Bnb4bitAdapter.matches(t):
        return Bnb4bitAdapter()
    if Nvfp4Adapter.matches(t):
        return Nvfp4Adapter()
    if Float8Adapter.matches(t):
        return Float8Adapter()
    if GgufAdapter.matches(t):
        return GgufAdapter()
    if RegularAdapter.matches(t):
        return RegularAdapter()
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
