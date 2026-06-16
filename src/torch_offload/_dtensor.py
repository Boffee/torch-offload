"""Internal optional-import module for PyTorch ``DTensor`` (tensor parallel).

Single source of truth for the small slice of the ``DTensor`` API that the
movement-only :class:`~torch_offload.dtensor_adapter.DTensorAdapter` needs:
the predicate, the local shard, and stable hashable signatures for the
``DeviceMesh`` and placements (used in identity / block-pool keys).

``DTensor`` is the standard PyTorch tensor-parallel representation: an outer
wrapper carrying a local shard plus a ``DeviceMesh`` and per-dim placements
(``Shard(dim)`` / ``Replicate()`` / ...). It is optional — absent when
``torch.distributed`` lacks the tensor API — so this module degrades to
"never matches" rather than importing at package load.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    from torch.distributed.tensor import DTensor

    DTENSOR_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    DTENSOR_AVAILABLE = False
    DTensor: Any = None


def is_dtensor(t: object) -> bool:
    """Return whether ``t`` is a PyTorch ``DTensor``."""
    return DTENSOR_AVAILABLE and isinstance(t, DTensor)


def require_dtensor(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated ``DTensor``, or raise."""
    if not is_dtensor(t):
        raise TypeError(f"expected a torch DTensor, got {type(t).__name__}")
    return t


def local_shard(t: Any) -> torch.Tensor:  # noqa: ANN401
    """The local shard of ``t`` if it is a ``DTensor``, else ``t`` unchanged.

    Tolerates the bare local shard that ``DTensorAdapter.cpu_param`` yields
    (a DTensor weight's resting representation is not a DTensor)."""
    return t.to_local() if is_dtensor(t) else t


def mesh_signature(
    mesh: Any,  # noqa: ANN401
    *,
    include_device: bool = True,
) -> tuple[object, ...]:
    """Identity-free structural signature of a ``DeviceMesh``.

    Captures device type, mesh shape, and the flattened global ranks — the
    fields that determine sharding compatibility — without any per-tensor
    identity. Stable across ranks so two equivalently-sharded weights key
    the same. ``include_device=False`` drops the device type, used by
    bind-layout validation where a weight's resting CPU mesh and resident
    CUDA mesh must compare equal (see :func:`cpu_mesh_for`)."""
    structural = (
        tuple(mesh.shape),
        tuple(int(r) for r in mesh.mesh.flatten().tolist()),
    )
    return (mesh.device_type, *structural) if include_device else structural


def cpu_mesh_for(mesh: Any) -> Any:  # noqa: ANN401
    """A CPU ``DeviceMesh`` mirroring ``mesh``'s shape and ranks.

    ``DTensorAdapter.cpu_param`` keeps the offloaded (resting) weight a DTensor
    on the host: ``from_local`` moves the local onto the mesh device, so a CUDA
    mesh would re-allocate GPU memory; a CPU mesh keeps the local on the host.
    Constructing a ``DeviceMesh`` initializes a (gloo) process group, but torch
    deduplicates the underlying group across equivalent meshes — so the adapter
    builds one per pinned param (held in its state) with no global cache."""
    from torch.distributed.device_mesh import DeviceMesh  # noqa: PLC0415

    return DeviceMesh("cpu", mesh.mesh.cpu())


def placements_key(placements: tuple[Any, ...]) -> tuple[str, ...]:
    """Hashable key for a placements tuple (``Shard(dim)`` / ``Replicate`` /
    ``Partial``). ``repr`` is stable across torch versions and distinguishes
    shard dimension."""
    return tuple(repr(p) for p in placements)


def rebuild_dtensor(
    local: torch.Tensor,
    mesh: Any,  # noqa: ANN401
    placements: Any,  # noqa: ANN401
    shape: Any,  # noqa: ANN401
    stride: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Rebuild a ``DTensor`` from an already-local shard plus its wrapper.

    ``run_check=False``: a pure local rewrap, never a collective. The local
    must already live on the mesh device (otherwise ``from_local`` moves it
    there, re-allocating). ``shape``/``stride`` are the original GLOBAL
    shape/stride — passed explicitly so ``from_local`` does not re-infer the
    global shape as ``local_size * world_size`` (wrong for uneven shards).
    ``mesh``/``placements`` are typed loosely here at the optional-import
    boundary."""
    return DTensor.from_local(
        local, mesh, placements, run_check=False, shape=shape, stride=stride
    )
