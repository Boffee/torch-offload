"""Per-buffer pinned-CPU storage primitive."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .tensor_adapters import clone_to_pinned_cpu


@dataclass(slots=True, eq=False)
class PinnedBuffer:
    """Pinned host storage for one registered buffer."""

    tensor: torch.Tensor

    @classmethod
    def clone(cls, buffer: torch.Tensor) -> PinnedBuffer:
        """Clone ``buffer`` into pinned CPU storage."""
        return cls(
            tensor=clone_to_pinned_cpu(
                buffer,
                memory_format=torch.contiguous_format,
            ),
        )

    @property
    def cache_bytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


__all__ = ["PinnedBuffer"]
