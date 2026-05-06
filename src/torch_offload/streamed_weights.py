"""Block-streaming primitive for memory-efficient training and inference.

A :class:`StreamedWeights` manages a single homogeneous block list:
pins the frozen weights to CPU at construction time, streams them
to GPU on demand via forward-pre hooks, and uses a pre-allocated
GPU slot pool plus a background prefetcher to overlap DMA with
compute.

This is the sharp, low-level primitive. It does NOT manage:

- Non-block parts of the model (parent-module state, sibling
  modules) — caller composes :class:`PinnedWeights` with the
  streamer's :attr:`slot_filter` for that.
- Trainable parameter movement — caller handles a separate
  :class:`~torch_offload.trainable_weights.TrainableWeights`.
- Cross-region tied-weight detection — that's a composer concern
  (see :func:`ModelOffloader` /
  :class:`~torch_offload.model_offloader.ModelOffloader`).

Most users want :func:`ModelOffloader` (the blessed safe
API). Reach for :class:`StreamedWeights` directly only when you need
bespoke composition (e.g., multiple block lists like Flux's
``transformer_blocks`` + ``single_transformer_blocks``).
"""

from __future__ import annotations

import contextlib
import functools
import logging
import weakref
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from types import TracebackType
from typing import Any

import torch
from torch import nn

from .pinned_buffer import PinnedParamBuffer
from .protocols import SlotOwnership
from .slots import iter_buffer_slots, iter_param_slots

logger = logging.getLogger(__name__)


def _release_cuda_cache_on_drop(is_cuda: bool) -> None:
    # Process-wide PyTorch CUDA allocator cache is the only state the
    # refcount-based GC of a streamer can't release on its own. Without
    # this, freed pinned/GPU pages stay held by the allocator until the
    # next allocation pressure event, which manifests as OOMs at
    # workload boundaries (e.g. successive trainers in one process).
    # ``empty_cache()`` is process-global (not per-device), so a single
    # bool is the right abstraction — capturing the device object would
    # imply per-device scoping that PyTorch doesn't actually provide.
    if not is_cuda:
        return
    # Finalizers can run at interpreter shutdown when CUDA is already torn
    # down, so suppress teardown-time noise.
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Pre-allocated GPU buffer pool
# ---------------------------------------------------------------------------


class _GpuSlot:
    """Pre-allocated GPU storage for one block's frozen params.

    For each ``PinnedParamBuffer`` template, allocates matching GPU
    tensors (data + optional scale for quanto) once at construction
    and builds a stable ``nn.Parameter`` wrapping each. Subsequent
    ``copy_from`` calls write the pinned bytes into the pre-allocated
    GPU tensors in place — no malloc on the hot path, and the
    Parameter wrappers stay identity-stable across loads (required
    for PEFT compatibility and to avoid Python ref churn).
    """

    __slots__ = ("_gpu_params", "_gpu_states")

    def __init__(self, template: list[PinnedParamBuffer], device: torch.device) -> None:
        # Opaque adapter-specific GPU state per buffer; the streamer
        # round-trips it through copy_to_gpu / make_gpu_param without
        # inspecting its shape.
        self._gpu_states: dict[str, Any] = {}
        self._gpu_params: dict[str, nn.Parameter] = {}
        for buf in template:
            gpu_state = buf.allocate_gpu_storage(device)
            self._gpu_states[buf.name] = gpu_state
            self._gpu_params[buf.name] = buf.make_gpu_param(gpu_state)

    def copy_from(self, bufs: list[PinnedParamBuffer], non_blocking: bool = False) -> None:
        for buf in bufs:
            buf.copy_to_gpu(self._gpu_states[buf.name], non_blocking=non_blocking)
            if buf.transform is not None:
                buf.transform.apply(self._gpu_params[buf.name].data)

    def get_param(self, name: str) -> nn.Parameter:
        return self._gpu_params[name]


class _GpuSlotPool:
    """Pool of pre-allocated :class:`_GpuSlot` instances."""

    def __init__(
        self,
        template: list[PinnedParamBuffer],
        num_slots: int,
        device: torch.device,
    ) -> None:
        self._slots = [_GpuSlot(template, device) for _ in range(num_slots)]
        self._free: list[int] = list(range(num_slots))
        self._events: list[torch.cuda.Event | None] = [None] * num_slots

    def acquire(self) -> int:
        return self._free.pop()

    def release(self, slot_id: int) -> None:
        self._free.append(slot_id)

    def slot(self, slot_id: int) -> _GpuSlot:
        return self._slots[slot_id]

    def set_compute_event(self, slot_id: int, event: torch.cuda.Event) -> None:
        self._events[slot_id] = event

    def wait_if_needed(self, slot_id: int, stream: torch.cuda.Stream | None) -> None:
        ev = self._events[slot_id]
        if ev is not None:
            if stream is not None and not ev.query():
                stream.wait_event(ev)
            self._events[slot_id] = None


# ---------------------------------------------------------------------------
# Block store: pinned CPU + GPU pool
# ---------------------------------------------------------------------------


class _BlockPinnedStore:
    """Per-block pinned CPU + (when activated) per-slot GPU storage.

    Two-phase construction:

    - ``__init__`` pins every frozen param/buffer into a fresh
      :class:`PinnedParamBuffer` / pinned clone, and records the slot
      locations these will eventually be installed at — but DOES NOT
      mutate the model's slots. This means a homogeneity check (or any
      other validation) can run on the templates and reject the
      configuration without leaving the model in a half-pinned state.
      If the store is dropped before :meth:`apply_slot_mutations`, the
      pinned tensors are released by GC.
    - :meth:`apply_slot_mutations` swaps the model's slots to point
      at the pinned objects. After this point the store owns slot
      state and :meth:`evict_block` is needed to restore CPU params.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        *,
        skip_slots: set[SlotOwnership] | None = None,
    ) -> None:
        self._layers = list(layers)
        self._param_bufs: list[list[PinnedParamBuffer]] = []
        self._param_locs: list[list[tuple[str, nn.Module, str]]] = []
        # Per-block list of (buffer_obj, parent_module, leaf_name, cpu_clone)
        # pending installation. apply_slot_mutations() does the
        # `mod_buf.data = cpu_clone` swaps.
        self._buf_records: list[
            list[tuple[torch.Tensor, nn.Module, str, torch.Tensor]]
        ] = []
        self._slots_applied = False
        skip: set[SlotOwnership] = skip_slots or set()
        # Slot-ownership filter built during the same walk: identifies
        # every (parent_module, leaf, kind) slot the streamer will own.
        # Stable across slot mutation, so consumers (PinnedWeights with
        # skip_slots) can reference it before OR after mutation.
        slot_filter: set[SlotOwnership] = set()

        for layer in self._layers:
            block_bufs: list[PinnedParamBuffer] = []
            block_locs: list[tuple[str, nn.Module, str]] = []
            # Slot filter covers every alias slot (so a composed
            # PinnedWeights skips them all). Pinning is deduped by id(p) —
            # intra-block aliasing is unsupported and must be rejected
            # upstream by detect_streaming_region_ties; if the caller
            # bypasses that, the alias slot silently keeps the original
            # Parameter on activate (the user's bug, not ours).
            seen_param_ids: set[int] = set()
            for s in iter_param_slots(layer):
                if s.slot in skip:
                    continue
                # Contract guard: streaming cycles slots through
                # pre-allocated requires_grad=False Parameter wrappers,
                # which destroys the per-Parameter identity that
                # optimizers and grad accumulation rely on. Trainable
                # slots must be partitioned out by the caller (composer
                # routes them via skip_slots; direct users are on the
                # hook). Fail loudly here rather than silently freezing.
                if s.param.requires_grad:
                    raise ValueError(
                        f"StreamedWeights cannot manage trainable slot {s.name!r}: "
                        "streaming swaps slot Parameters with frozen pool "
                        "wrappers, breaking optimizer identity. Use "
                        "ModelOffloader (which partitions trainables "
                        "into TrainableWeights automatically), or pass the "
                        "slot in skip_slots and route it to a separate "
                        "trainable mover."
                    )
                slot_filter.add(s.slot)
                if id(s.param) in seen_param_ids:
                    continue
                seen_param_ids.add(id(s.param))
                block_bufs.append(PinnedParamBuffer(s.name, s.param))
                block_locs.append((s.name, s.parent, s.leaf))
            self._param_bufs.append(block_bufs)
            self._param_locs.append(block_locs)

            buf_records: list[tuple[torch.Tensor, nn.Module, str, torch.Tensor]] = []
            seen_buffer_ids: set[int] = set()
            for s in iter_buffer_slots(layer):
                if s.slot in skip:
                    continue
                slot_filter.add(s.slot)
                if id(s.buffer) in seen_buffer_ids:
                    continue
                seen_buffer_ids.add(id(s.buffer))
                cpu_clone = s.buffer.data.clone(memory_format=torch.contiguous_format).pin_memory()
                buf_records.append((s.buffer, s.parent, s.leaf, cpu_clone))
            self._buf_records.append(buf_records)

        self._slot_filter: frozenset[SlotOwnership] = frozenset(slot_filter)
        self._device: torch.device | None = None
        self._pool: _GpuSlotPool | None = None
        self._block_to_slot: dict[int, int] = {}
        self._pool_config: tuple[int, torch.device] | None = None

    @property
    def slot_filter(self) -> frozenset[SlotOwnership]:
        """``SlotOwnership`` set covering every slot the store manages.
        Stable across the store's lifetime."""
        return self._slot_filter

    def apply_slot_mutations(self) -> None:
        """Install the pinned cpu_params + buffer clones into the
        model's slots. Idempotent."""
        if self._slots_applied:
            return
        for block_bufs, block_locs in zip(
            self._param_bufs, self._param_locs, strict=True,
        ):
            for buf, (_qn, submod, local_name) in zip(
                block_bufs, block_locs, strict=True,
            ):
                # _parameters[leaf] swap (rather than .data) is required
                # for quanto correctness — see PinnedWeights for details.
                submod._parameters[local_name] = buf.cpu_param
        for buf_records in self._buf_records:
            for mod_buf, _submod, _leaf, cpu_clone in buf_records:
                mod_buf.data = cpu_clone
        self._slots_applied = True

    @property
    def cache_bytes(self) -> int:
        total = 0
        for block in self._param_bufs:
            for buf in block:
                total += buf.cache_bytes
        for buf_records in self._buf_records:
            for _mb, _sm, _ln, cpu_clone in buf_records:
                total += cpu_clone.numel() * cpu_clone.element_size()
        return total

    def is_homogeneous(self) -> bool:
        if len(self._param_bufs) <= 1:
            return True
        ref = self._param_bufs[0]

        # Per-block layout signature: each buffer's name (slot identity)
        # paired with its adapter-provided homogeneity_key (storage
        # shape/dtype/stride/quant-metadata, whatever the adapter
        # considers layout-significant).
        def _key(b: PinnedParamBuffer) -> tuple:
            return (b.name, b.homogeneity_key)

        ref_keys = [_key(b) for b in ref]
        for block in self._param_bufs[1:]:
            if len(block) != len(ref_keys):
                return False
            for ref_tup, b in zip(ref_keys, block, strict=True):
                if _key(b) != ref_tup:
                    return False
        return True

    def activate_pool(self, num_gpu_slots: int, device: torch.device) -> None:
        if self._pool_config is not None:
            existing = self._pool_config
            if existing != (num_gpu_slots, device):
                raise ValueError(
                    f"_BlockPinnedStore pool already activated with "
                    f"{existing}; cannot re-activate with ({num_gpu_slots}, "
                    f"{device}). Call deactivate_pool() first."
                )
            return
        self._device = device
        self._pool_config = (num_gpu_slots, device)
        if num_gpu_slots > 0 and self._param_bufs and self.is_homogeneous():
            self._pool = _GpuSlotPool(self._param_bufs[0], num_gpu_slots, device)

    def deactivate_pool(self) -> None:
        self._pool = None
        self._block_to_slot.clear()
        self._pool_config = None

    def load_block(
        self,
        idx: int,
        device: torch.device,
        non_blocking: bool = False,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if self._pool is not None:
            self._load_pooled(idx, non_blocking, stream)
        else:
            self._load_alloc(idx, device, non_blocking)

    def _load_pooled(self, idx: int, non_blocking: bool, stream: torch.cuda.Stream | None) -> None:
        slot_id = self._block_to_slot.get(idx)
        if slot_id is None:
            slot_id = self._pool.acquire()
            self._block_to_slot[idx] = slot_id
        self._pool.wait_if_needed(slot_id, stream)
        slot = self._pool.slot(slot_id)
        slot.copy_from(self._param_bufs[idx], non_blocking=non_blocking)

        for qual_name, submod, local_name in self._param_locs[idx]:
            submod._parameters[local_name] = slot.get_param(qual_name)
        for mod_buf, _sm, _ln, cpu_clone in self._buf_records[idx]:
            mod_buf.data = cpu_clone.to(self._device, non_blocking=non_blocking)

    def _load_alloc(self, idx: int, device: torch.device, non_blocking: bool) -> None:
        for buf, (_qn, submod, local_name) in zip(
            self._param_bufs[idx], self._param_locs[idx], strict=True,
        ):
            submod._parameters[local_name] = buf.load_to_gpu(device, non_blocking=non_blocking)
        for mod_buf, _sm, _ln, cpu_clone in self._buf_records[idx]:
            mod_buf.data = cpu_clone.to(device, non_blocking=non_blocking)

    def evict_block_fast(self, idx: int) -> None:
        if self._pool is None:
            # Without a pool there's no slot to release — restore the
            # CPU param to drop the model's reference to GPU storage
            # so it can be reclaimed by refcount.
            return self.evict_block(idx)
        slot_id = self._block_to_slot.pop(idx, None)
        if slot_id is not None:
            self._pool.release(slot_id)

    def evict_block(self, idx: int) -> None:
        for buf, (_qn, submod, local_name) in zip(
            self._param_bufs[idx], self._param_locs[idx], strict=True,
        ):
            submod._parameters[local_name] = buf.cpu_param
        for mod_buf, _sm, _ln, cpu_clone in self._buf_records[idx]:
            mod_buf.data = cpu_clone
        if self._pool is not None:
            slot_id = self._block_to_slot.pop(idx, None)
            if slot_id is not None:
                self._pool.release(slot_id)

    def mark_compute_done(self, idx: int, event: torch.cuda.Event) -> None:
        if self._pool is not None:
            slot_id = self._block_to_slot.get(idx)
            if slot_id is not None:
                self._pool.set_compute_event(slot_id, event)


# ---------------------------------------------------------------------------
# LRU tracker
# ---------------------------------------------------------------------------


class _BlockTracker:
    def __init__(self) -> None:
        self._on_gpu: set[int] = set()
        self._lru: OrderedDict[int, None] = OrderedDict()
        self.peak_gpu_blocks = 0

    def is_on_gpu(self, idx: int) -> bool:
        return idx in self._on_gpu

    def touch(self, idx: int) -> None:
        if idx in self._lru:
            self._lru.move_to_end(idx)

    def mark_on_gpu(self, idx: int) -> None:
        self._on_gpu.add(idx)
        self._lru.pop(idx, None)
        self._lru[idx] = None

    def mark_on_cpu(self, idx: int) -> None:
        self._on_gpu.discard(idx)
        self._lru.pop(idx, None)

    def pick_victim(self, protected: set[int]) -> int:
        for idx in self._lru:
            if idx not in protected:
                return idx
        raise RuntimeError("no evictable block")

    def clear(self) -> None:
        self._on_gpu.clear()
        self._lru.clear()


# ---------------------------------------------------------------------------
# StreamedWeights — public block-streaming primitive
# ---------------------------------------------------------------------------


class StreamedWeights:
    """Streams a single block list between pinned CPU and GPU.

    The sharp, low-level streaming primitive. Manages frozen weights
    of the block list ONLY: pins them to CPU at construction time,
    streams them to GPU via forward-pre hooks on :meth:`activate`,
    releases GPU resources on :meth:`deactivate`. Does not touch
    parent modules, sibling modules, or trainable parameters — those
    are the composer's responsibility.

    A :class:`StreamedWeights` is a *component* meant to be composed
    inside a :class:`~torch_offload.model_offloader.ModelOffloader`.
    It deliberately does NOT implement
    :class:`~torch_offload.protocols.ModelStrategy` (its
    :meth:`activate` returns ``None`` because it doesn't own the
    model). For top-level use, build a strategy via
    :func:`~torch_offload.model_offloader.ModelOffloader`.

    Lifecycle is uniform with :class:`PinnedWeights`: ``__init__``
    pins (so ``cache_bytes`` is final at construction time, ready
    for :class:`~torch_offload.model_cache.ModelCache` admission),
    ``activate`` brings to GPU, ``deactivate`` returns slots to
    pinned CPU and removes hooks. There is no ``close()``; pinned
    memory in module slots is freed when the caller drops the
    strategy and model references.

    **Calling deactivate() before dropping the strategy is preferred**
    — it removes the forward hooks (cleaner model state) and reverts
    GPU-resident blocks back to pinned CPU. The hook closure uses
    ``weakref.ref(self)`` so dropping the strategy without deactivate
    is non-fatal: the orphaned hooks remain installed on the model
    but no-op; resident blocks stay on GPU until the model itself is
    dropped. Forward calls through still-resident blocks work; calls
    through previously-evicted blocks find pinned-CPU slots (slow but
    functional).

    Parameters
    ----------
    blocks:
        The resolved sequence of block modules. Caller is responsible
        for path resolution. Typically an ``nn.ModuleList``.
    target_device:
        GPU device to stream to.
    blocks_to_swap:
        Number of blocks to keep offloaded on CPU at any time. Must
        be ``< len(blocks)``.
    prefetch_count:
        How many blocks ahead to prefetch on a background thread.
    cyclic:
        Default ``False``. When ``True``, treat the block list as a
        cyclic sequence: large index jumps (``|Δidx| > num_blocks/2``)
        are interpreted as iteration wraparound rather than direction
        reversal, and prefetch indices wrap modulo ``num_blocks``.
        Suitable for inference loops that iterate the model repeatedly
        (diffusion denoising, multi-step decoders) — the prefetcher
        keeps streaming the next iteration's leading blocks instead
        of misfiring at iteration boundaries. Leave ``False`` for
        single-shot or genuinely non-cyclic traversals.

        The wraparound heuristic assumes monotonic intra-iteration
        traversal (each iteration walks blocks in order, forward or
        reverse, possibly skipping by one or two). Non-monotonic
        forward jumps larger than ``num_blocks/2`` within a single
        iteration would be misclassified as wraparound. The flag is
        captured at :meth:`activate` time; flipping ``self._cyclic``
        on a live streamer has no effect until the next
        deactivate/activate cycle.
    name:
        Optional human-readable label for log messages.
    strict_homogeneous:
        Default ``True``. When True, raises ``ValueError`` if the
        blocks are not layout-identical (param names/shapes/dtypes/
        quanto specs differ). The check runs before any slot
        mutation, so failure leaves the user's model untouched.
        When False, falls back to per-load ``cudaMalloc`` allocation
        — slow but works for heterogeneous configurations. Use
        multiple :class:`StreamedWeights`s (one per homogeneous group)
        with :func:`ModelOffloader` /
        :class:`ModelOffloader` to get the pool benefit on
        heterogeneous models like Flux.
    skip_slots:
        Optional set of :class:`SlotOwnership` tuples identifying
        ``(parent_module, leaf, kind)`` slots inside the blocks that
        the streamer should not pin. Used by composers to route
        trainable in-block params (LoRA / PEFT adapters) to a separate
        strategy. Streaming cannot host trainables — slot replacement
        breaks Parameter identity. A trainable slot not in
        ``skip_slots`` triggers a contract-guard ValueError at
        construction; this is intentional, fail-loud over silent
        freezing of the param.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        target_device: torch.device,
        *,
        blocks_to_swap: int,
        prefetch_count: int = 2,
        cyclic: bool = False,
        name: str | None = None,
        strict_homogeneous: bool = True,
        skip_slots: set[SlotOwnership] | None = None,
    ) -> None:
        self._blocks: list[nn.Module] = list(blocks)
        self._target_device = target_device
        self._blocks_to_swap = blocks_to_swap
        self._prefetch_count = prefetch_count
        self._cyclic = cyclic
        self._name = name or f"StreamedWeights({len(self._blocks)} blocks)"

        if blocks_to_swap >= len(self._blocks):
            raise ValueError(
                f"blocks_to_swap ({blocks_to_swap}) must be < num blocks ({len(self._blocks)})"
            )

        # Pin in __init__ — uniform lifecycle with PinnedWeights, and
        # ModelCache integration sees a final `cache_bytes` immediately.
        # Two-phase under the hood: build pinned templates → validate
        # homogeneity → apply slot mutations. If the homogeneity check
        # fails, the local `store` binding falls out of scope, the
        # pinned tensors are GC'd, and the user's model is left
        # untouched.
        for block in self._blocks:
            block.to("cpu")
        store = _BlockPinnedStore(
            self._blocks, skip_slots=skip_slots,
        )
        if strict_homogeneous and not store.is_homogeneous():
            raise ValueError(
                f"{self._name}: blocks are not homogeneous (different "
                "param names/shapes/dtypes/quanto specs across blocks). "
                "Either pass strict_homogeneous=False to opt into the "
                "slower per-load allocation fallback, or split into "
                "multiple StreamedWeights instances — one per homogeneous group — "
                "and compose with ModelOffloader()."
            )
        store.apply_slot_mutations()
        self._store: _BlockPinnedStore | None = store

        # Active resources allocated on activate(); presence of
        # ``_executor`` is the de-facto "active" indicator.
        self._tracker: _BlockTracker | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._executor: ThreadPoolExecutor | None = None
        self._stream: torch.cuda.Stream | None = None
        self._pending: dict[int, Future[None]] = {}
        self._prefetch_events: dict[int, torch.cuda.Event] = {}
        self._last_idx: int = -1

        # Auto-flush the CUDA allocator cache when the streamer is GC'd,
        # so callers don't need to remember an explicit empty_cache() at
        # workload boundaries. Captures only a bool (no self ref) so it
        # never blocks collection.
        weakref.finalize(
            self, _release_cuda_cache_on_drop, target_device.type == "cuda"
        )

    @property
    def slot_filter(self) -> frozenset[SlotOwnership]:
        """``SlotOwnership`` set covering every (parent, leaf, kind)
        slot the streamer owns. Stable across the streamer's
        lifetime — safe to read at any point and survives slot
        mutation, so a consumer can construct a
        :class:`PinnedWeights` with this filter regardless of order
        relative to the streamer."""
        assert self._store is not None
        return self._store.slot_filter

    @property
    def param_bufs_per_block(self) -> list[list[PinnedParamBuffer]]:
        """Per-block lists of :class:`PinnedParamBuffer` objects.

        Used by :class:`~torch_offload.ModelOffloader` to build a
        reverse index from parameter qualified names to their buffers.

        .. warning::
           Returned list aliases internal store state. Treat it as
           read-only — mutating the outer list, inner lists, or
           buffer objects will corrupt streaming bookkeeping.
        """
        assert self._store is not None
        return self._store._param_bufs

    @property
    def param_locs_per_block(self) -> list[list[tuple[str, nn.Module, str]]]:
        """Per-block lists of ``(qualified_name, parent_module, leaf_name)``.

        Parallel structure to :attr:`param_bufs_per_block`. Used by the
        composer to build a reverse index from parameter names to
        parent modules — needed by routed-mode LoRA (forward hooks) and
        any future mechanism that has to act on the layer object rather
        than the weight buffer.

        .. warning::
           Returned list aliases internal store state. Treat it as
           read-only — mutating it will corrupt streaming bookkeeping.
        """
        assert self._store is not None
        return self._store._param_locs

    @property
    def cache_bytes(self) -> int:
        return self._store.cache_bytes if self._store is not None else 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """Allocate per-activation resources (GPU slot pool, CUDA
        stream/events, prefetch executor, forward hooks). The
        composite's :meth:`activate` returns the model — this method
        returns ``None`` because the streamer doesn't own one.

        **Lifecycle is caller's responsibility.** Calling activate()
        twice without an intervening deactivate() will register
        forward hooks twice, double-stream every block — undefined
        behavior. Don't.

        **Failure semantics (poison-on-failure):** if activation
        fails midway, the streamer is left in an undefined state with
        partial resources allocated. The caller's only valid next
        action is :meth:`deactivate`, which idempotently tears down
        whatever was allocated."""
        assert self._store is not None

        num_layers = len(self._blocks)
        num_resident = num_layers - self._blocks_to_swap
        num_gpu_slots = num_resident + self._prefetch_count

        self._tracker = _BlockTracker()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._stream = torch.cuda.Stream(device=self._target_device, priority=-1)
        self._pending = {}
        self._prefetch_events = {i: torch.cuda.Event() for i in range(num_layers)}
        self._last_idx = -1

        self._store.activate_pool(num_gpu_slots, self._target_device)

        for idx in range(min(num_resident, num_layers)):
            self._store.load_block(idx, self._target_device)
            self._tracker.mark_on_gpu(idx)

        self._register_hooks(num_resident)
        self.reset_peak()

        logger.info(
            f"{self._name} active: {self._blocks_to_swap}/{num_layers} on CPU, "
            f"{num_resident} resident on GPU, prefetch={self._prefetch_count}, "
            f"gpu_pool_slots={num_gpu_slots}"
        )

    def deactivate(self) -> None:
        """Tear down active resources idempotently — safe to call
        before activate or multiple times. Every step in
        ``_teardown_active_resources`` null-checks; partial state
        from a failed activate cleans up correctly. Drop the
        strategy reference after deactivate to release pinned
        memory."""
        prefetch_exc = self._teardown_active_resources()
        if prefetch_exc is not None:
            raise prefetch_exc

    def __enter__(self) -> None:
        self.activate()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.deactivate()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def peak_gpu_blocks(self) -> int:
        return self._tracker.peak_gpu_blocks if self._tracker is not None else 0

    def reset_peak(self) -> None:
        if self._tracker is not None:
            self._tracker.peak_gpu_blocks = len(self._tracker._on_gpu) + len(self._pending)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _teardown_active_resources(self) -> BaseException | None:
        """Idempotent cleanup of all active resources. Returns the
        first prefetch exception encountered (or None) so the caller
        can surface it after cleanup completes."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        # Drain prefetcher first: shutdown(wait=True) blocks until all
        # submitted futures complete (success OR fail). Then we use the
        # non-raising future.exception() to inspect each — no per-future
        # try/except needed.
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        first_prefetch_exc: BaseException | None = None
        for idx, future in self._pending.items():
            exc = future.exception()
            if exc is not None and first_prefetch_exc is None:
                first_prefetch_exc = exc
            if self._tracker is not None:
                self._tracker.mark_on_gpu(idx)
        self._pending.clear()
        self._prefetch_events.clear()

        if self._stream is not None:
            self._stream.synchronize()
            self._stream = None

        if self._tracker is not None and self._store is not None:
            torch.cuda.synchronize(device=self._target_device)
            for idx in range(len(self._blocks)):
                self._store.evict_block(idx)
            self._tracker.clear()

        if self._store is not None:
            self._store.deactivate_pool()

        self._tracker = None
        self._last_idx = -1

        return first_prefetch_exc

    def _evict_one(
        self, protected: set[int], compute_event: torch.cuda.Event | None = None
    ) -> None:
        victim = self._tracker.pick_victim(protected=protected)
        if compute_event is not None:
            self._store.mark_compute_done(victim, compute_event)
        self._store.evict_block_fast(victim)
        self._tracker.mark_on_cpu(victim)

    def _do_prefetch(self, idx: int) -> None:
        with torch.cuda.stream(self._stream):
            self._store.load_block(idx, self._target_device, non_blocking=True, stream=self._stream)
            self._prefetch_events[idx].record(self._stream)

    def _submit_prefetch(self, idx: int, max_on_gpu: int) -> None:
        if idx < 0 or idx >= len(self._blocks):
            return
        if self._tracker.is_on_gpu(idx) or idx in self._pending:
            return
        if len(self._tracker._on_gpu) + len(self._pending) >= max_on_gpu:
            return
        self._pending[idx] = self._executor.submit(self._do_prefetch, idx)

    def _ensure_on_gpu(self, idx: int) -> None:
        future = self._pending.pop(idx, None)
        if future is not None:
            future.result()
            ev = self._prefetch_events[idx]
            if not ev.query():
                torch.cuda.current_stream(self._target_device).wait_event(ev)
            self._tracker.mark_on_gpu(idx)
            return

        if not self._tracker.is_on_gpu(idx):
            self._store.load_block(idx, self._target_device)
            self._tracker.mark_on_gpu(idx)

    def _register_hooks(self, num_resident: int) -> None:
        idx_map: dict[int, int] = {id(layer): idx for idx, layer in enumerate(self._blocks)}
        prefetch_count = self._prefetch_count  # capture as local — no `self` ref in closure
        max_on_gpu = num_resident + prefetch_count
        cyclic = self._cyclic
        num_layers = len(self._blocks)
        wrap_threshold = num_layers // 2  # |Δidx| > this counts as wraparound
        # weakref breaks the cycle: layer → hook → closure → streamer
        # → _blocks → layer. Without it, dropping the strategy without
        # first calling deactivate() would keep everything alive until
        # Python's cycle collector runs (not refcount-based GC). The
        # weak ref lets refcount immediately free the strategy when the
        # caller drops it; orphaned hooks no-op safely.
        self_ref = weakref.ref(self)

        def _pre_hook(_module: nn.Module, _args: Any, *, idx: int) -> None:  # noqa: ANN401
            streamer = self_ref()
            if streamer is None:
                # Strategy was dropped without deactivate. Hook is
                # orphaned — no-op. Forward through this block uses
                # whatever the slot currently holds: GPU param if it
                # was resident at drop-time, pinned cpu_param if it
                # had been evicted. Both are functional (CPU forward
                # is slower but works on pinned tensors).
                return
            tracker = streamer._tracker
            pending = streamer._pending
            if tracker.is_on_gpu(idx):
                tracker.touch(idx)
            else:
                compute_event = torch.cuda.current_stream(streamer._target_device).record_event()
                while len(tracker._on_gpu) >= num_resident:
                    protected = {idx} | set(pending.keys())
                    streamer._evict_one(protected, compute_event)
                streamer._ensure_on_gpu(idx)

            # Direction inference. In cyclic mode, a large index jump
            # (|Δ| > num_layers/2) is iteration wraparound, not a
            # reversal: keep the same forward/backward sense and let
            # prefetch indices wrap modulo num_layers so the next
            # iteration's leading blocks get streamed proactively.
            last = streamer._last_idx
            streamer._last_idx = idx
            if last < 0:
                direction = 1
            else:
                diff = idx - last
                if cyclic and abs(diff) > wrap_threshold:
                    direction = -1 if diff > 0 else 1
                else:
                    direction = 1 if diff >= 0 else -1
            for offset in range(1, prefetch_count + 1):
                target = idx + direction * offset
                if cyclic:
                    target %= num_layers
                streamer._submit_prefetch(target, max_on_gpu)

            total = len(tracker._on_gpu) + len(pending)
            tracker.peak_gpu_blocks = max(tracker.peak_gpu_blocks, total)

        for layer in self._blocks:
            idx = idx_map[id(layer)]
            h = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            self._hooks.append(h)


__all__ = [
    "StreamedWeights",
]
