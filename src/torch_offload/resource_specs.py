"""Standard :class:`ResourceSpec` implementations.

The cache itself remains resource-agnostic. These frozen dataclasses adapt
model, LoRA, and ordinary-object factories to the structural resource-spec
protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import torch
from torch import nn

from .lora import LoRA
from .model_offloader import ModelOffloader
from .protocols import ResourceStore

M = TypeVar("M", bound=nn.Module)
T = TypeVar("T")


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelSpec(Generic[M]):
    """Model resource built from one user model factory.

    ``factory`` runs once to construct the cached :class:`ModelOffloader`.
    Every lease reuses that same model runtime sequentially; overlapping uses
    are rejected by the offloader.
    """

    key: str
    estimated_cache_bytes: int
    factory: Callable[[], M]
    blocks_attr: tuple[str, ...] = ()
    stream_trainable_weights: bool = False

    def build_store(self) -> ModelOffloader:
        """Build, pin, and bind the cached model runtime."""
        return ModelOffloader.from_module(
            self.factory(),
            blocks_attr=self.blocks_attr,
            stream_trainable_weights=self.stream_trainable_weights,
        )

    def value(self, store: ResourceStore) -> ModelOffloader:
        """Return the leased model runtime."""
        return cast(ModelOffloader, store)


@dataclass(frozen=True, kw_only=True, slots=True)
class LoRASpec:
    """LoRA resource built from a state-dict factory.

    ``blocks_attr`` and ``dtype`` are forwarded to
    :meth:`LoRA.from_state_dict`. Pass the base model's block paths so
    routed mode can co-schedule factor blocks with that model.
    """

    key: str
    estimated_cache_bytes: int
    factory: Callable[[], dict[str, torch.Tensor]]
    blocks_attr: tuple[str, ...] = ()
    dtype: torch.dtype | None = None

    def build_store(self) -> LoRA:
        """Build and pin this reusable adapter resource."""
        return LoRA.from_state_dict(
            self.factory(),
            blocks_attr=self.blocks_attr,
            dtype=self.dtype,
        )

    def value(self, store: ResourceStore) -> LoRA:
        """Return the leased LoRA resource."""
        return cast(LoRA, store)


@dataclass(frozen=True, slots=True)
class _ObjectStore(Generic[T]):
    """Accounting wrapper for a plain Python object."""

    value: T
    cache_bytes: int


@dataclass(frozen=True, kw_only=True, slots=True)
class ObjectSpec(Generic[T]):
    """Resource spec for a tokenizer, processor, config, or other object.

    Every lease yields the same object instance. The default zero-byte charge
    keeps ordinary heap objects outside the pinned-host-memory budget.
    """

    key: str
    factory: Callable[[], T]
    estimated_cache_bytes: int = 0

    def build_store(self) -> ResourceStore:
        """Build the accounting wrapper around the cached object."""
        return _ObjectStore(self.factory(), self.estimated_cache_bytes)

    def value(self, store: ResourceStore) -> T:
        """Return the object held by its accounting store."""
        return cast(_ObjectStore[T], store).value


__all__ = ["LoRASpec", "ModelSpec", "ObjectSpec"]
