"""Identity-preserving mover for trainable parameters.

:class:`TrainableWeights` is a :class:`ModelStrategyComponent` that
brings trainable params (LoRA adapters, PEFT layers) to GPU on
:meth:`activate` and back to CPU on :meth:`deactivate`, preserving
``nn.Parameter`` object identity so optimizer state survives.
"""

from __future__ import annotations

from types import TracebackType

import torch
from torch import nn

__all__ = ["TrainableWeights"]


class TrainableWeights:
    """Strategy component for the model's trainable parameters.

    The trainable counterpart to :class:`PinnedWeights`. Both components
    bring their managed params to the target device on
    :meth:`activate` and return them to CPU on :meth:`deactivate`,
    but the mechanisms are mirror images:

    - :class:`PinnedWeights` owns pinned-CPU clones, slot-replaces the
      Parameter wrapper at every transition. Frozen-only — slot
      replacement orphans optimizer state.
    - :class:`TrainableWeights` owns nothing (``cache_bytes=0``); the
      user's Parameter objects stay alive in their slots, and only
      ``p.data`` storage moves via ``p.data = p.data.to(device)``.
      Identity-preserving — optimizer state survives.

    Walks ``model.parameters()`` each transition (deduped by Parameter
    identity), so the standard ``tie_weights()`` pattern (one Parameter
    aliased at multiple slots) is handled correctly. Distinct-Parameter
    tied storage is rejected upstream by
    :func:`detect_streaming_region_ties` because moving each Parameter
    independently would untie the alias on GPU.
    """

    def __init__(self, model: nn.Module, target_device: torch.device) -> None:
        self._model = model
        self._target_device = target_device

    @property
    def cache_bytes(self) -> int:
        return 0

    def activate(self) -> None:
        self._move(self._target_device)

    def deactivate(self) -> None:
        self._move(torch.device("cpu"))

    def _move(self, device: torch.device) -> None:
        for p in self._model.parameters():
            if p.requires_grad:
                if p.data.device != device:
                    p.data = p.data.to(device)
                if p.grad is not None and p.grad.device != device:
                    p.grad = p.grad.to(device)

    def __enter__(self) -> None:
        self.activate()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.deactivate()
