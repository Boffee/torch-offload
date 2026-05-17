"""Shared pinned group records."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .pinned_param import PinnedParam
from .slots import BufferSlot, ParamSlot, unique_slots


@dataclass(slots=True)
class PinnedParamGroup:
    pinned: PinnedParam
    slots: list[ParamSlot]

    @property
    def unique_slots(self) -> list[ParamSlot]:
        return unique_slots(self.slots)


@dataclass(slots=True)
class PinnedBufferGroup:
    pinned: torch.Tensor
    slots: list[BufferSlot]

    @property
    def unique_slots(self) -> list[BufferSlot]:
        return unique_slots(self.slots)


__all__ = [
    "PinnedBufferGroup",
    "PinnedParamGroup",
]
