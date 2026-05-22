"""Identity-preserving mover for trainable parameters.

:class:`TrainableWeights` is a :class:`ModelStrategyComponent` that
brings trainable params (LoRA adapters, PEFT layers) to GPU on
:meth:`activate` and back to CPU on :meth:`deactivate`, preserving
``nn.Parameter`` object identity so optimizer state survives.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

from ._devices import canonical_device
from .protocols import SlotKey
from .slots import iter_param_slots, set_param_data

__all__ = ["TrainableWeights"]


class TrainableWeights:
    """Strategy component for the model's trainable parameters.

    A zero-cache counterpart to :class:`PinnedWeights` for trainables
    that do not need a pinned CPU clone. Both components bring their
    managed params to the activation device on :meth:`activate` and
    return them to CPU on :meth:`deactivate`, but the ownership model
    differs:

    - :class:`PinnedWeights` owns pinned-CPU clones and uses an explicit
      optimizer-step copy-back boundary for CUDA trainable updates.
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
    skip_slots:
        Optional set of :class:`SlotKey` values identifying
        ``(parent_module, leaf, kind)`` slots to skip — used by
        composers that route some trainables to a different mover
        (e.g., :class:`ModelOffloader` routes in-block trainables to
        the streamer's per-block ``.data``-swap path and only hands
        out-of-block trainables to ``TrainableWeights``).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        skip_slots: set[SlotKey] | None = None,
    ) -> None:
        self._model = model
        self._skip_slots = frozenset(skip_slots) if skip_slots is not None else None

    @property
    def cache_bytes(self) -> int:
        return 0

    def activate(self, device: torch.device | str | None = None) -> None:
        self._move(self._resolve_device(device))

    def deactivate(self) -> None:
        self._move(torch.device("cpu"))

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        if device is not None:
            return canonical_device(device)
        raise ValueError(
            "TrainableWeights.activate() requires a device"
        )

    def _move(self, device: torch.device) -> None:
        for p in self._iter_trainable_params():
            if not p.requires_grad:
                continue
            if p.data.device != device:
                set_param_data(p, p.data.to(device))
            if p.grad is not None and p.grad.device != device:
                p.grad = p.grad.to(device)

    def _iter_trainable_params(self) -> Iterator[nn.Parameter]:
        if self._skip_slots is None:
            yield from self._model.parameters()
            return

        seen_ids: set[int] = set()
        for s in iter_param_slots(self._model):
            if s.key in self._skip_slots:
                continue
            param = s.get()
            if not param.requires_grad:
                continue
            if id(param) in seen_ids:
                continue
            seen_ids.add(id(param))
            yield param
