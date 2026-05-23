"""Explicit tensor adapter selection and storage identity.

The adapter classes themselves live in ``tensor_adapters.py`` and the
format-specific modules. This module is the one normal Python place that
knows the built-in dispatch order and adapter-defined storage identity.
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


def storage_key(t: torch.Tensor) -> tuple[Any, ...]:
    """Identity key for tied-weight detection."""
    return select_adapter(t).storage_key(t)


def param_storage_key(param: nn.Parameter) -> tuple[Any, ...]:
    """Return a storage-identity grouping key for a parameter."""
    if param.numel() == 0:
        # Zero-sized tensors all share data_ptr()==0; key by object
        # identity so aliases of the same Parameter still dedupe.
        return ("__empty__", id(param))
    return storage_key(param.data)


def buffer_storage_key(buffer: torch.Tensor) -> tuple[Any, ...]:
    """Return a storage-identity grouping key for a registered buffer."""
    if buffer.numel() == 0:
        return ("__empty_buf__", id(buffer))
    return storage_key(buffer)


__all__ = [
    "buffer_storage_key",
    "param_storage_key",
    "select_adapter",
    "storage_key",
]
