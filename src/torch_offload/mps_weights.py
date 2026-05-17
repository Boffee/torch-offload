"""Minimal whole-model CPU->MPS materializer."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch import nn

from ._devices import canonical_device
from .slots import (
    assert_frozen,
    iter_buffer_slots,
    iter_param_slots,
)

__all__ = ["MpsWeights"]


class MpsWeights:
    """Materialize a CPU model on MPS one tensor slot at a time.

    This is intentionally just the strategy lifecycle around a simple
    constructor-time copy-and-replace loop. It does not keep a second CPU
    cache and does not try to preserve more advanced invariants such as
    tied parameter aliases. ``activate()`` and ``deactivate()`` are
    lifecycle no-ops; the model stays on MPS.
    """

    def __init__(self, model: nn.Module, include_buffers: bool = True) -> None:
        self._model = model
        self._include_buffers = include_buffers
        self._cache_bytes = self._validate_and_count()
        if self._cache_bytes == 0:
            raise ValueError(
                "MpsWeights requires at least one frozen parameter or, "
                "when include_buffers=True, at least one registered buffer "
                "to manage."
            )
        self._check_mps_available()
        self._move_to_mps()
        self._synchronize_mps()

    @property
    def cache_bytes(self) -> int:
        """Managed model bytes, counted from the construction-time module."""

        return self._cache_bytes

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def value(self) -> nn.Module:
        return self._model

    def activate(self, device: torch.device | str | None = None) -> None:
        if device is None:
            raise ValueError(
                "MpsWeights.activate() requires device='mps'; pass "
                "activate('mps') or use this strategy through "
                "ModelCache.use(..., device='mps')"
            )
        active_device = canonical_device(device)
        if active_device.type != "mps":
            raise ValueError(
                "MpsWeights.activate() supports MPS; "
                f"got {active_device}."
            )

    def deactivate(self) -> None:
        return

    @contextlib.contextmanager
    def use(self, device: torch.device | str) -> Iterator[nn.Module]:
        self.activate(device)
        try:
            yield self._model
        finally:
            self.deactivate()

    def _validate_and_count(self) -> int:
        total = 0
        for slot in iter_param_slots(self._model):
            assert_frozen(slot, owner="MpsWeights")
            param = slot.get()
            self._require_cpu(param.data, slot.name)
            total += param.numel() * param.element_size()
        if self._include_buffers:
            for slot in iter_buffer_slots(self._model):
                buffer = slot.get()
                self._require_cpu(buffer, slot.name)
                total += buffer.numel() * buffer.element_size()
        return total

    def _move_to_mps(self) -> None:
        device = torch.device("mps")
        for slot in iter_param_slots(self._model):
            param = slot.get()
            if param.requires_grad:
                raise RuntimeError("MpsWeights is frozen-only, but a managed parameter became trainable.")
            if param.device != device:
                slot.set(
                    nn.Parameter(
                        self._copy_tensor(param.detach(), device),
                        requires_grad=False,
                    ),
                )

        if self._include_buffers:
            for slot in iter_buffer_slots(self._model):
                buffer = slot.get()
                if buffer.device == device:
                    continue
                slot.set(
                    self._copy_tensor(buffer.detach(), device),
                )

    def _copy_tensor(self, source: torch.Tensor, device: torch.device) -> torch.Tensor:
        return source.to(device)

    @staticmethod
    def _require_cpu(tensor: torch.Tensor, name: str) -> None:
        if tensor.device.type != "cpu":
            raise ValueError(
                f"MpsWeights requires CPU tensors at construction; "
                f"{name!r} is on {tensor.device}."
            )

    @staticmethod
    def _check_mps_available() -> None:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError(
                "MpsWeights requires an available PyTorch MPS backend."
            )

    @staticmethod
    def _synchronize_mps() -> None:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            torch.mps.synchronize()
