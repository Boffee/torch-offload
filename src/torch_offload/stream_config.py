"""Runtime block-streaming policy, supplied at activation.

:class:`StreamConfig` carries the knobs that govern GPU residency and
prefetch behaviour for a single activation: how many blocks stay resident,
how far ahead to prefetch, and whether the block sequence is cyclic.

These are deliberately *not* part of the pinned backing store. They do not
affect what is pinned to host or the cache budget (``cache_bytes``), so they
ride :meth:`activate` (or :meth:`ModelCache.use`) rather than store
construction — the value flows straight to where it is consumed, with no
component state to fall out of sync with a running stream.
``num_resident_blocks`` is clamped to the block count at activation, so one
config stays valid across models of different depth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """Block-streaming residency/prefetch policy for one activation.

    Attributes
    ----------
    num_resident_blocks:
        Maximum number of streamed blocks resident on GPU at a time. Must be
        ``>= 1``; values above the block count clamp to it at activation. ``1``
        is right for almost all workloads: eviction is LRU, so a sequential scan
        reloads every block each pass regardless of residency — extra resident
        slots cost GPU memory without reducing transfer volume. The exception is
        checkpointed training, whose backward recompute reverses direction and
        finds the last ``num_resident_blocks`` blocks still resident.
    num_prefetch_blocks:
        How many blocks ahead to prefetch on a background thread. At most
        ``num_resident_blocks + num_prefetch_blocks`` blocks are GPU-resident at
        once; for a homogeneous block list the GPU target pool holds exactly
        that many targets (a heterogeneously quantized list keeps one reusable
        target per distinct block-layout signature). Spare VRAM is usually
        better spent here than on extra resident blocks.
    cyclic:
        When ``True``, treat the block list as a cyclic sequence: large index
        jumps (``|Δidx| > num_blocks/2``) are read as iteration wraparound
        rather than direction reversal, and prefetch indices wrap modulo
        ``num_blocks``. Suitable for inference loops that iterate the model
        repeatedly (diffusion denoising, multi-step decoders); leave ``False``
        for single-shot traversals. The heuristic assumes monotonic
        intra-iteration traversal.
    """

    num_resident_blocks: int = 1
    num_prefetch_blocks: int = 2
    cyclic: bool = False

    def __post_init__(self) -> None:
        # num_resident_blocks > num_blocks is allowed and clamps at
        # activation, so one config stays valid across models of
        # different depths.
        if self.num_resident_blocks < 1:
            raise ValueError(f"num_resident_blocks ({self.num_resident_blocks}) must be >= 1")
        if self.num_prefetch_blocks < 0:
            raise ValueError(f"num_prefetch_blocks ({self.num_prefetch_blocks}) must be >= 0")


#: Default policy used when ``activate`` is called without a ``stream_config``
#: (one resident block, two prefetched, linear traversal). A shared immutable —
#: not per-instance state.
DEFAULT_STREAM_CONFIG = StreamConfig()


__all__ = ["DEFAULT_STREAM_CONFIG", "StreamConfig"]
