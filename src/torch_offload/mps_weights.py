"""Minimal whole-model CPU->MPS materializer."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch import nn

from ._devices import canonical_device
from .module_names import (
    named_buffer_entries,
    named_parameter_entries,
    set_named_buffer,
    set_named_parameter,
)

__all__ = ["MpsWeights"]


class MpsWeights:
    """Materialize a CPU model on MPS one named tensor at a time.

    This is intentionally just the binding lifecycle around a simple
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

    def bind(self) -> MpsWeights:
        return self

    def activate(
        self, device: torch.device | str | None = None, **kwargs: object,
    ) -> None:
        del kwargs  # MPS materialization takes no streaming policy
        if device is None:
            raise ValueError(
                "MpsWeights.activate() requires device='mps'; pass "
                "activate('mps') or use this binding through "
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
        for name, param in self._model.named_parameters(remove_duplicate=False):
            _assert_frozen_param(name, param)
            self._require_cpu(param.data, name)
            total += param.numel() * param.element_size()
        if self._include_buffers:
            for name, buffer in self._model.named_buffers(remove_duplicate=False):
                self._require_cpu(buffer, name)
                total += buffer.numel() * buffer.element_size()
        return total

    def _move_to_mps(self) -> None:
        device = torch.device("mps")
        for _name, parent, leaf, param in named_parameter_entries(self._model):
            if param.requires_grad:
                raise RuntimeError(
                    "MpsWeights is frozen-only, but a managed parameter "
                    "became trainable."
                )
            if param.device != device:
                set_named_parameter(
                    parent,
                    leaf,
                    nn.Parameter(
                        self._copy_tensor(param.detach(), device),
                        requires_grad=False,
                    ),
                )

        if self._include_buffers:
            for _name, parent, leaf, buffer, persistent in named_buffer_entries(
                self._model,
            ):
                if buffer.device == device:
                    continue
                set_named_buffer(
                    parent,
                    leaf,
                    self._copy_tensor(buffer.detach(), device),
                    persistent=persistent,
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


def _assert_frozen_param(name: str, param: nn.Parameter) -> None:
    if not param.requires_grad:
        return
    raise ValueError(
        f"MpsWeights cannot manage trainable parameter {name!r}: "
        "replacing the Parameter object would break optimizer/grad "
        "identity."
    )
