"""Internal helpers for normalizing user-provided torch devices."""

from __future__ import annotations

import torch


def canonical_device(device: torch.device | str) -> torch.device:
    """Return a stable torch.device for storage and equality checks.

    ``torch.device("cuda")`` is intentionally indexless: PyTorch resolves it
    against the current CUDA device when an operation runs. Cache leases and
    streaming state need stable identity, so bind bare CUDA to the current
    CUDA device at acquire/activation time. CPU is normalized to its
    indexless form because it is a pass-through host-backed placement.
    """
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return torch.device("cpu")
    if (
        resolved.type == "cuda"
        and resolved.index is None
        and torch.cuda.is_available()
    ):
        return torch.device("cuda", torch.cuda.current_device())
    return resolved
