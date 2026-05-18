"""Shared pinned binding records."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .pinned_param import PinnedParam
from .slots import BufferSlot, ParamSlot, unique_slots


@dataclass(slots=True)
class PinnedParamBinding:
    """One model instance's parameter slots bound to one pinned backing."""

    pinned: PinnedParam
    slots: list[ParamSlot]
    cpu_param: nn.Parameter

    @property
    def unique_slots(self) -> list[ParamSlot]:
        return unique_slots(self.slots)


@dataclass(slots=True)
class PinnedBufferBinding:
    """One model instance's buffer slots bound to one pinned tensor."""

    pinned: torch.Tensor
    slots: list[BufferSlot]

    @property
    def unique_slots(self) -> list[BufferSlot]:
        return unique_slots(self.slots)


__all__ = [
    "PinnedBufferBinding",
    "PinnedParamBinding",
]
