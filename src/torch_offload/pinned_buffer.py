"""Per-buffer pinned-CPU storage primitive."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .tensor_adapters import clone_to_pinned_cpu


@dataclass(slots=True, eq=False)
class PinnedBuffer:
    """Pinned host storage for one registered buffer.

    ``name`` is the module-relative buffer name used to identify the
    corresponding target storage slot. It comes from PyTorch-style
    ``named_buffers()`` paths and mirrors ``PinnedParam.name`` for
    parameter storage.
    """

    name: str
    tensor: torch.Tensor

    @classmethod
    def clone(cls, name: str, buffer: torch.Tensor) -> PinnedBuffer:
        """Clone ``buffer`` into pinned CPU storage."""
        return cls(
            name=name,
            tensor=clone_to_pinned_cpu(
                buffer,
                memory_format=torch.contiguous_format,
            ),
        )

    @property
    def cache_bytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


__all__ = ["PinnedBuffer"]
