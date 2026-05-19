"""Explicit tensor adapter selection.

The adapter classes themselves live in ``tensor_adapters.py`` and the
format-specific modules. This module is the one normal Python place
that knows the built-in dispatch order.
"""

from __future__ import annotations

from typing import Any

import torch

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


__all__ = [
    "select_adapter",
    "storage_key",
]
