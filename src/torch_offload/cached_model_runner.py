"""Composition layer for cached model and LoRA resources.

The cache owns resource admission and leases; this runner owns LoRA attachment
and device activation. Keeping those responsibilities separate lets
:class:`ResourceCache` remain unaware of models and adapters.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import TypeVar, cast

import torch
from torch import nn

from .lora import LoRA, LoraMode
from .model_offloader import ModelOffloader
from .resource_cache import ResourceCache
from .resource_specs import LoRASpec, ModelSpec
from .stream_config import StreamConfig

M = TypeVar("M", bound=nn.Module)


class CachedModelRunner:
    """Activate cached model runtimes with leased LoRA dependencies.

    A model cache entry owns one :class:`ModelOffloader` and supports
    sequential reuse only. Its lifecycle rejects overlapping uses regardless
    of how many runners share the cache.
    """

    def __init__(self, cache: ResourceCache) -> None:
        self._cache = cache

    @contextlib.contextmanager
    def use(
        self,
        model: ModelSpec[M],
        *,
        device: torch.device | str,
        lora_specs: Sequence[LoRASpec] = (),
        lora_strengths: Sequence[float] | None = None,
        lora_mode: LoraMode = "merge",
        stream_config: StreamConfig | None = None,
    ) -> Iterator[M]:
        """Lease dependencies and activate the cached model runtime.

        ``lora_strengths`` defaults to one for each LoRA and, when supplied,
        must have the same length as ``lora_specs``. A LoRA resource key may
        appear only once in a use.
        """
        specs = tuple(lora_specs)
        strengths = None if lora_strengths is None else tuple(lora_strengths)
        if strengths is not None and len(strengths) != len(specs):
            raise ValueError(
                "lora_strengths must have the same length as lora_specs"
            )
        if len({spec.key for spec in specs}) != len(specs):
            raise ValueError(
                "lora_specs must not contain the same LoRA resource key more than once"
            )

        # Dependencies are leased first, so admitting the model cannot evict a
        # LoRA selected for this same runtime.
        with self._cache.lease_many((*specs, model)) as resources:
            loras = cast(tuple[LoRA, ...], resources[:-1])
            offloader = cast(ModelOffloader, resources[-1])
            config = stream_config if stream_config is not None else StreamConfig()
            offloader.activate(
                device,
                loras=loras,
                lora_strengths=strengths,
                lora_mode=lora_mode,
                stream_config=config,
            )
            try:
                yield cast(M, offloader.value)
            finally:
                offloader.deactivate()


__all__ = ["CachedModelRunner"]
