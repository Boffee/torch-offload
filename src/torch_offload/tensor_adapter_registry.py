"""Explicit tensor adapter selection and tensor identity.

The adapter classes themselves live in ``tensor_adapters.py`` and the
format-specific modules. This module is the one normal Python place that
knows the built-in dispatch order and adapter-defined tensor identity.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .gguf_adapter import GgufAdapter
from .nvfp4_adapter import Nvfp4Adapter
from .quanto_adapter import QuantoAdapter
from .tensor_adapters import RegularAdapter, TensorAdapter


def select_adapter(t: torch.Tensor) -> TensorAdapter[Any, Any]:
    """Return the built-in adapter that handles ``t``."""
    if QuantoAdapter.matches(t):
        return QuantoAdapter()
    if Nvfp4Adapter.matches(t):
        return Nvfp4Adapter()
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


def param_tensor_id(param: nn.Parameter) -> tuple[Any, ...]:
    """Return an adapter-defined tensor identity for a parameter."""
    if param.numel() == 0:
        # Zero-sized tensors all share data_ptr()==0; key by object
        # identity so aliases of the same Parameter still dedupe.
        return ("__empty__", id(param))
    return tensor_id(param.data)


def buffer_tensor_id(buffer: torch.Tensor) -> tuple[Any, ...]:
    """Return an adapter-defined tensor identity for a registered buffer."""
    if buffer.numel() == 0:
        return ("__empty_buf__", id(buffer))
    return tensor_id(buffer)


__all__ = [
    "buffer_tensor_id",
    "param_tensor_id",
    "select_adapter",
    "tensor_id",
]
