"""Identity-preserving mover for trainable parameters.

:class:`TrainableWeights` is a :class:`ModelStrategyComponent` that
brings trainable params (LoRA adapters, PEFT layers) to GPU on
:meth:`activate` and back to CPU on :meth:`deactivate`, preserving
``nn.Parameter`` object identity so optimizer state survives.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import TracebackType

import torch
from torch import nn

from .protocols import SlotOwnership
from .slots import iter_param_slots

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

    Each transition walks the model and moves every currently
    trainable Parameter. With ``skip_slots`` the walk is slot-aware,
    which lets :class:`ModelOffloader` route in-block streamed
    trainable weights to the streamer while leaving out-of-block
    trainables here. Keeping the walk dynamic preserves the historical
    behavior for late ``requires_grad`` changes.

    Parameters
    ----------
    model:
        The model whose trainable params should be moved.
    target_device:
        GPU device to move to on :meth:`activate`.
    skip_slots:
        Optional set of :class:`SlotOwnership` tuples identifying
        ``(parent_module, leaf, kind)`` slots to skip — used by
        composers that route some trainables to a different mover
        (e.g., :class:`ModelOffloader` routes in-block trainables to
        the streamer's per-block ``.data``-swap path and only hands
        out-of-block trainables to ``TrainableWeights``).
    """

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        *,
        skip_slots: set[SlotOwnership] | None = None,
    ) -> None:
        self._model = model
        self._target_device = target_device
        self._skip_slots = frozenset(skip_slots) if skip_slots is not None else None

    @property
    def cache_bytes(self) -> int:
        return 0

    def activate(self) -> None:
        self._move(self._target_device)

    def deactivate(self) -> None:
        self._move(torch.device("cpu"))

    def _move(self, device: torch.device) -> None:
        for p in self._iter_trainable_params():
            if not p.requires_grad:
                continue
            if p.data.device != device:
                p.data = p.data.to(device)
            if p.grad is not None and p.grad.device != device:
                p.grad = p.grad.to(device)

    def _iter_trainable_params(self) -> Iterator[nn.Parameter]:
        if self._skip_slots is None:
            yield from self._model.parameters()
            return

        seen_ids: set[int] = set()
        for s in iter_param_slots(self._model):
            if s.slot in self._skip_slots:
                continue
            if not s.param.requires_grad:
                continue
            if id(s.param) in seen_ids:
                continue
            seen_ids.add(id(s.param))
            yield s.param

    def __enter__(self) -> None:
        self.activate()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.deactivate()
