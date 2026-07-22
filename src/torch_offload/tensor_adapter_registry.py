"""Tensor adapter registration, selection, and tensor identity.

The adapter classes themselves live in ``tensor_adapters.py`` and the
format-specific modules. This module is the one normal Python place that
knows the dispatch order and adapter-defined tensor identity. Downstream
tensor subclasses can participate through :func:`register_adapter` without
adding a format-specific dependency to this package.
"""

from __future__ import annotations

from collections.abc import Callable
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
from .static_float8_adapter import StaticFloat8Adapter
from .tensor_adapters import RegularAdapter, TensorAdapter

# DTensorAdapter is always checked before registered and built-in adapters: a
# DTensor is an outer wrapper whose local shard is delegated back through this
# registry. External adapters therefore compose inside DTensor without
# replacing its distributed wrapper behavior.

# Remaining built-in adapters in dispatch order; first match wins.
# RegularAdapter is last — it matches only exact torch.Tensor/nn.Parameter, so
# structured subclasses reach their dedicated adapter first.
_BUILTIN_ADAPTERS: tuple[type[TensorAdapter[Any, Any]], ...] = (
    QuantoAdapter,
    Bnb4bitAdapter,
    Bnb8bitAdapter,
    Nvfp4Adapter,
    MxAdapter,
    StaticFloat8Adapter,
    Float8Adapter,
    Int8Adapter,
    Int4TilePackedAdapter,
    GgufAdapter,
    RegularAdapter,
)

# Process-global external registrations. New adapters are prepended so a later,
# more-specific registration can override an earlier broad match. Registration
# is intended for application initialization, before models are constructed.
_REGISTERED_ADAPTERS: list[type[TensorAdapter[Any, Any]]] = []


def register_adapter(
    adapter_cls: type[TensorAdapter[Any, Any]],
) -> Callable[[], None]:
    """Register a stateless external tensor adapter.

    External adapters are checked newest-first, after :class:`DTensorAdapter`
    and before all other built-ins. ``adapter_cls`` must satisfy the
    :class:`TensorAdapter` protocol and be constructible without arguments.

    Registration is process-global and intended for application startup,
    before constructing models or pinned resources. Returns an idempotent
    callable that removes this registration, primarily for tests and scoped
    integrations.
    """
    if not isinstance(adapter_cls, type):
        raise TypeError(
            "register_adapter() expects a TensorAdapter class, "
            f"got {type(adapter_cls).__name__}"
        )
    if adapter_cls is DTensorAdapter or adapter_cls in _BUILTIN_ADAPTERS:
        raise ValueError(
            f"TensorAdapter {adapter_cls.__name__!r} is already built in"
        )
    if adapter_cls in _REGISTERED_ADAPTERS:
        raise ValueError(
            f"TensorAdapter {adapter_cls.__name__!r} is already registered"
        )

    try:
        adapter = adapter_cls()
    except TypeError as exc:
        raise TypeError(
            f"TensorAdapter {adapter_cls.__name__!r} must be constructible "
            "without arguments"
        ) from exc
    if not isinstance(adapter, TensorAdapter):
        raise TypeError(
            f"Adapter {adapter_cls.__name__!r} does not satisfy the "
            "TensorAdapter protocol"
        )

    _REGISTERED_ADAPTERS.insert(0, adapter_cls)
    removed = False

    def remove_adapter() -> None:
        nonlocal removed

        if removed:
            return
        _REGISTERED_ADAPTERS.remove(adapter_cls)
        removed = True

    return remove_adapter


def select_adapter(t: torch.Tensor) -> TensorAdapter[Any, Any]:
    """Return the highest-priority adapter that handles ``t``."""
    if DTensorAdapter.matches(t):
        return DTensorAdapter()
    for adapter_cls in _REGISTERED_ADAPTERS:
        if adapter_cls.matches(t):
            return adapter_cls()
    for adapter_cls in _BUILTIN_ADAPTERS:
        if adapter_cls.matches(t):
            return adapter_cls()
    raise NotImplementedError(
        f"No TensorAdapter for tensor type {type(t).__name__!r}. "
        "Plain tensors are handled by RegularAdapter; tensor subclasses "
        "need a dedicated adapter registered with register_adapter()."
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
    "register_adapter",
    "select_adapter",
    "tensor_id",
]
