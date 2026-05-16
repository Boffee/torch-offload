"""Block-streaming primitive for memory-efficient training and inference.

A :class:`StreamedWeights` manages a single block list whose blocks
share the same parameter layout (names, shapes, dtypes, and any
quanto/GGUF wrapper metadata): pins the params to CPU at
construction time, streams them to GPU on demand via forward-pre
hooks, and uses a pre-allocated GPU slot pool plus a background
prefetcher to overlap DMA with compute. On CPU, the host-backed
pinned state is used directly without streaming. Heterogeneous block lists
(e.g. Flux's two block kinds) split into multiple
:class:`StreamedWeights` instances composed via :class:`ModelOffloader`.

In-block trainable params (LoRA adapters) flow through the same slot
pool; ``_BlockPinnedStore`` branches on ``buf.requires_grad`` to swap
``.data`` (preserves user Parameter identity for autograd / optimizer
state) instead of replacing the Parameter wrapper. Gradients live on
GPU during backward via PyTorch's native ``AccumulateGrad``; only
``.data`` is materialized around ``optimizer.step()`` via
:meth:`optimizer_step`.

This is the sharp, low-level primitive. It does NOT manage:

- Non-block parts of the model (parent-module state, sibling
  modules) — caller composes :class:`PinnedWeights` with the
  streamer's :attr:`slot_filter` for that.
- Out-of-block trainable parameter movement — caller handles a
  separate :class:`~torch_offload.trainable_weights.TrainableWeights`.
- Cross-region tied-weight detection — that's a composer concern
  (see :func:`ModelOffloader` /
  :class:`~torch_offload.model_offloader.ModelOffloader`).
- Activation-checkpointing enforcement — required for in-block
  trainable streaming, but checked at the composer level.

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
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import cast

import torch
from torch import nn

from ._devices import canonical_device
from .pinned_buffer import PinnedParamBuffer
from .protocols import SlotOwnership
from .slots import iter_buffer_slots, iter_param_slots
from .tensor_adapters import RegularAdapter

logger = logging.getLogger(__name__)

_ParamSpec = tuple[str, nn.Parameter, nn.Module, str]
_ParamAliasSpec = list[tuple[str, nn.Module, str]]
_BufferSpec = tuple[torch.Tensor, nn.Module, str]


def _repoint_data_to_pinned(
    param: nn.Parameter, buf: PinnedParamBuffer
) -> None:
    """ExitStack callback used by :meth:`StreamedWeights.optimizer_step`.

    Repoints ``param.data`` at the pinned host clone, releasing the
    optimizer-step GPU allocation that was assigned to ``param.data``
    on enter. Defined at module scope so the closure registered with
    ``ExitStack.callback`` doesn't capture a streamer reference.
    """
    param.data = buf.cpu_param.data


def _slot_param(parent: nn.Module, leaf: str) -> nn.Parameter:
    param = parent._parameters[leaf]
    if param is None:
        raise RuntimeError(f"Parameter slot {leaf!r} is unexpectedly empty")
    return param


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
        self._gpu_states: dict[str, object] = {}
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


def _layout_signature(p: nn.Parameter) -> tuple:
    """Layout fields that block 0's pool template must match across blocks.

    ``Tensor.copy_`` silently casts dtype and silently broadcasts
    compatible shapes — both invisible failure modes that would
    silently corrupt a load. Wrapper metadata (qtype, axis,
    activation_qtype, quant_type) is similarly invisible to copy_.

    For plain tensors, stride is normalized to contiguous by
    ``clone_pin`` so it's not part of the signature — including it
    would falsely reject transposed inputs that pinning normalizes.
    For subclassed wrappers (quanto, GGUF), the wrapper's logical
    stride is captured by ``clone_pin`` into ``state.stride`` and
    survives pinning (the GPU param is rebuilt with that stride),
    so it IS load-bearing and goes in the signature. Inner storage
    shapes/dtypes (``_data``, ``_scale``) are also captured for
    subclassed wrappers — they're mostly determined by the other
    fields, but explicit is cheap and forecloses any wrapper-class
    edge case.
    """
    t = p.data
    parts: list = [tuple(t.shape), t.dtype]
    for attr in ("qtype", "axis", "activation_qtype", "quant_type"):
        if hasattr(t, attr):
            parts.append((attr, getattr(t, attr)))
    if type(t) is not torch.Tensor:
        parts.append(("wrapper_stride", tuple(t.stride())))
        for inner_attr in ("_data", "_scale"):
            inner = getattr(t, inner_attr, None)
            if inner is not None:
                parts.append((inner_attr, tuple(inner.shape), inner.dtype))
    return tuple(parts)


def _check_block_layouts_match(
    param_specs: list[list[_ParamSpec]],
) -> None:
    """Raise if blocks have mismatched param layouts. Called before
    pinning so failures leave the model untouched.

    See :func:`_layout_signature` for what counts as "matched."
    """
    if len(param_specs) <= 1:
        return

    def sig(specs: list[_ParamSpec]) -> tuple:
        return tuple((name, _layout_signature(param)) for name, param, _, _ in specs)

    ref = sig(param_specs[0])
    for i in range(1, len(param_specs)):
        if sig(param_specs[i]) != ref:
            raise ValueError(
                f"Block {i} param layout differs from block 0. "
                "All blocks in a StreamedWeights group must share the "
                "same param structure (names, shapes, dtypes, and any "
                "quanto/GGUF wrapper metadata). Split heterogeneous "
                "block lists across separate `layers_attr=[...]` "
                "groups in ModelOffloader."
            )


def _collect_block_slot_specs(
    layers: list[nn.Module],
    skip: set[SlotOwnership],
) -> tuple[
    list[list[_ParamSpec]],
    list[list[_ParamAliasSpec]],
    list[list[_BufferSpec]],
    frozenset[SlotOwnership],
]:
    param_specs: list[list[_ParamSpec]] = []
    param_alias_specs: list[list[_ParamAliasSpec]] = []
    buffer_specs: list[list[_BufferSpec]] = []
    slot_filter: set[SlotOwnership] = set()

    for layer in layers:
        block_params: list[_ParamSpec] = []
        block_aliases_by_id: dict[int, _ParamAliasSpec] = {}
        block_param_order: list[int] = []
        seen_param_ids: set[int] = set()
        for s in iter_param_slots(layer):
            if s.slot in skip:
                continue
            slot_filter.add(s.slot)
            param_id = id(s.param)
            aliases = block_aliases_by_id.get(param_id)
            if aliases is None:
                aliases = []
                block_aliases_by_id[param_id] = aliases
                block_param_order.append(param_id)
            aliases.append((s.name, s.parent, s.leaf))
            if param_id in seen_param_ids:
                continue
            seen_param_ids.add(param_id)
            block_params.append((s.name, s.param, s.parent, s.leaf))
        param_specs.append(block_params)
        param_alias_specs.append(
            [block_aliases_by_id[param_id] for param_id in block_param_order]
        )

        block_bufs: list[_BufferSpec] = []
        seen_buffer_ids: set[int] = set()
        for s in iter_buffer_slots(layer):
            if s.slot in skip:
                continue
            slot_filter.add(s.slot)
            if id(s.buffer) in seen_buffer_ids:
                continue
            seen_buffer_ids.add(id(s.buffer))
            block_bufs.append((s.buffer, s.parent, s.leaf))
        buffer_specs.append(block_bufs)

    return param_specs, param_alias_specs, buffer_specs, frozenset(slot_filter)


class _BlockPinnedStore:
    """Per-block pinned CPU + (when activated) per-slot GPU storage.

    Construction is three-phase so an invalid configuration **does
    not pin and does not mutate the model's slot identities**:

    1. ``__init__`` first walks each block to collect param/buffer
       slot specs (no pinning) and verifies that every block shares
       the same layout signature. A mismatch raises ``ValueError``
       before any pin or slot mutation.
    2. Then it pins every managed param/buffer into a fresh
       :class:`PinnedParamBuffer` / pinned clone and records the
       slot locations they'll be installed at. ``__init__`` does NOT
       install those pinned objects into the model's slots.
    3. :meth:`apply_slot_mutations` swaps the model's slots to point
       at the pinned objects. After this point the store owns slot
       state and :meth:`evict_block` is needed to restore CPU params.

    Scope of the "no pin / no slot mutation" guarantee on
    validation failure: callers (``StreamedWeights.__init__`` and
    ``ModelOffloader.__init__``) move the model to CPU *before*
    invoking this constructor — that placement change is not
    undone by the validator. The guarantee covers ``pin_memory()``
    and ``submod._parameters[leaf] = ...`` mutations only.

    Note also that ``PinnedParamBuffer.__init__`` opportunistically
    repoints the user's Parameter ``.data`` at the pinned clone as a
    memory optimization (see ``pinned_buffer.py``). That mutation
    happens during phase 2 and isn't undoable, but phase 1 has
    already validated the configuration so it only fires for
    configs that are about to succeed.
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
        self._param_aliases: list[list[list[tuple[str, nn.Module, str]]]] = []
        # Per-block list of (buffer_obj, parent_module, leaf_name, cpu_clone)
        # pending installation. apply_slot_mutations() does the
        # `mod_buf.data = cpu_clone` swaps.
        self._buf_records: list[
            list[tuple[torch.Tensor, nn.Module, str, torch.Tensor]]
        ] = []
        self._slots_applied = False
        skip: set[SlotOwnership] = skip_slots or set()

        # Pass 1: walk each block to collect param/buffer slot specs
        # WITHOUT pinning anything. Pinning runs in pass 2 only after
        # the layout-signature check passes, so invalid configurations
        # raise before any model mutation.
        (
            param_specs,
            param_alias_specs,
            buffer_specs,
            self._slot_filter,
        ) = _collect_block_slot_specs(self._layers, skip)

        # Validate before pinning. ``Tensor.copy_`` silently casts dtype
        # and silently broadcasts compatible shapes, so any block N with
        # mismatched dtype, name, or wrapper metadata would otherwise
        # load into block 0's pool slot without raising and corrupt
        # forward. Run before pinning so an invalid config leaves the
        # model untouched.
        _check_block_layouts_match(param_specs)

        # Pass 2: pin params + buffers.
        for block_params, block_aliases in zip(
            param_specs, param_alias_specs, strict=True,
        ):
            block_pinned: list[PinnedParamBuffer] = []
            block_locs: list[tuple[str, nn.Module, str]] = []
            for name, param, parent, leaf in block_params:
                buf = PinnedParamBuffer(name, param)
                # Trainable streaming uses ``.data`` swap, which doesn't
                # preserve quanto's subclass wrapper on assignment.
                # Trainable quantized weights also aren't a real
                # workload (PyTorch can't easily train through quanto).
                # Reject at construction rather than failing later.
                if buf.requires_grad and buf.adapter is not RegularAdapter:
                    raise NotImplementedError(
                        f"Trainable streaming requires plain "
                        f"torch.Tensor params; slot {name!r} uses "
                        f"{buf.adapter.__name__}. Quantized weights "
                        f"are inference-only — keep them frozen, or "
                        f"wrap with PEFT/LoRA so the trainable "
                        f"adapter is plain-tensor."
                    )
                block_pinned.append(buf)
                block_locs.append((name, parent, leaf))
            self._param_bufs.append(block_pinned)
            self._param_locs.append(block_locs)
            self._param_aliases.append(block_aliases)

        for block_bufs in buffer_specs:
            buf_records: list[tuple[torch.Tensor, nn.Module, str, torch.Tensor]] = []
            for buf, parent, leaf in block_bufs:
                cpu_clone = buf.data.clone(memory_format=torch.contiguous_format).pin_memory()
                buf_records.append((buf, parent, leaf, cpu_clone))
            self._buf_records.append(buf_records)

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
        model's slots. Idempotent.

        Frozen params: ``submod._parameters[leaf] = buf.cpu_param``
        replaces the slot Parameter with a fresh
        ``Parameter(requires_grad=False)`` wrapping the pinned host
        clone. This orphans the user's pre-pin Parameter — fine for
        frozen, since there's no autograd/optimizer state to lose.

        Trainable params: leave the user's Parameter in place so
        autograd and optimizer references survive. The
        ``PinnedParamBuffer`` constructor already repointed the user
        Parameter's ``.data`` at the pinned clone (plain-tensor
        memory optimization), so the original storage is released
        without slot mutation.
        """
        if self._slots_applied:
            return
        for block_bufs, block_locs in zip(
            self._param_bufs, self._param_locs, strict=True,
        ):
            for buf, (_qn, submod, local_name) in zip(
                block_bufs, block_locs, strict=True,
            ):
                if buf.requires_grad:
                    # Trainable: keep user's Parameter, .data is already pinned.
                    continue
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
        if num_gpu_slots <= 0:
            raise ValueError(
                f"num_gpu_slots must be > 0, got {num_gpu_slots}. "
                "num_resident is always >= 1 by construction; this only "
                "fires when prefetch_count is negative."
            )
        self._device = device
        self._pool_config = (num_gpu_slots, device)
        # Pool template comes from block 0. The constructor's layout
        # check has already verified every other block matches.
        self._pool = _GpuSlotPool(self._param_bufs[0], num_gpu_slots, device)

    def deactivate_pool(self) -> None:
        self._pool = None
        self._block_to_slot.clear()
        self._pool_config = None

    def load_block(
        self,
        idx: int,
        non_blocking: bool = False,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        assert self._pool is not None, "activate_pool not called"
        slot_id = self._block_to_slot.get(idx)
        if slot_id is None:
            slot_id = self._pool.acquire()
            self._block_to_slot[idx] = slot_id
        self._pool.wait_if_needed(slot_id, stream)
        slot = self._pool.slot(slot_id)
        slot.copy_from(self._param_bufs[idx], non_blocking=non_blocking)

        for buf, (qual_name, submod, local_name) in zip(
            self._param_bufs[idx], self._param_locs[idx], strict=True,
        ):
            if buf.requires_grad:
                # Trainable: .data swap into the slot's GPU storage.
                # Preserves the user's Parameter object identity — so
                # autograd and optimizer state survive across cycles.
                # Reuses ``slot.get_param`` purely for its ``.data``
                # storage; the wrapper itself isn't installed.
                _slot_param(submod, local_name).data = slot.get_param(qual_name).data
            else:
                submod._parameters[local_name] = slot.get_param(qual_name)
        for mod_buf, _sm, _ln, cpu_clone in self._buf_records[idx]:
            mod_buf.data = cpu_clone.to(self._device, non_blocking=non_blocking)

    def validate_cpu_activation_supported(self) -> None:
        if any(
            buf.transform is not None
            for block_bufs in self._param_bufs
            for buf in block_bufs
        ):
            raise ValueError(
                "StreamedWeights transforms require CUDA activation; "
                "got cpu."
            )

    def release_slot(self, idx: int) -> None:
        """Return ``idx``'s pool slot for reuse.

        Releasing a block always restores every managed slot in that
        block to its pinned CPU representation before the pool slot is
        reusable. This is the core residency invariant: a non-resident
        block never leaves module slots pointing at reusable GPU scratch
        storage.
        """
        self.evict_block(idx)

    def evict_block(self, idx: int) -> None:
        """Restore ``idx``'s slot Parameters to their pinned CPU
        forms and release the pool slot. Used for teardown so the
        model can be safely accessed without the streamer.

        Frozen: ``submod._parameters[leaf] = buf.cpu_param``.
        Trainable: ``submod._parameters[leaf].data = buf.cpu_param.data``
        (preserve the user's Parameter object).
        """
        for buf, (_qn, submod, local_name) in zip(
            self._param_bufs[idx], self._param_locs[idx], strict=True,
        ):
            if buf.requires_grad:
                _slot_param(submod, local_name).data = buf.cpu_param.data
            else:
                submod._parameters[local_name] = buf.cpu_param
        for mod_buf, _sm, _ln, cpu_clone in self._buf_records[idx]:
            mod_buf.data = cpu_clone
        if self._pool is not None:
            slot_id = self._block_to_slot.pop(idx, None)
            if slot_id is not None:
                self._pool.release(slot_id)

    def evict_allocated_blocks(self) -> None:
        """Evict every block that currently holds a pool slot.

        ``_block_to_slot`` is the source of truth for slot allocation:
        a block is recorded there as soon as :meth:`load_block`
        (foreground or prefetch) acquires its slot, and removed only
        by :meth:`evict_block` / :meth:`release_slot`. Iterating it
        catches both currently-resident blocks and pending-prefetch
        blocks whose H2D may still be in flight — so callers (teardown,
        optimizer_step) don't have to reconcile multiple bookkeeping sources
        (tracker, pending dict, etc.).

        Caller is responsible for stream/event synchronization before
        the eviction so in-flight DMA into the slot bytes has settled.
        """
        for idx in list(self._block_to_slot.keys()):
            self.evict_block(idx)

    def move_trainable_grads_to(self, device: torch.device) -> None:
        """Move each trainable's ``.grad`` (if any) to ``device``.

        During backward, PyTorch's native ``AccumulateGrad`` writes
        grads on the param's data device — which is GPU at that point
        because the ``.data`` swap in :meth:`load_block` repointed
        ``.data`` at slot storage. The streamer's ``.data`` lifecycle
        restores ``.data`` to pinned CPU on eviction, but ``.grad``
        keeps living wherever AccumulateGrad placed it.

        This is called on activate to move any previously-deactivated
        CPU grads back to GPU before accumulation, and on deactivate
        to honor the documented "model on CPU" contract.
        """
        for _buf, parent, leaf in self.iter_trainables():
            param = _slot_param(parent, leaf)
            if param.grad is not None and param.grad.device != device:
                moved = param.grad.to(device)
                if param.data.device == device:
                    param.grad = moved
                else:
                    # PyTorch's grad setter rejects cross-device grad/data
                    # pairs. Streamed trainables intentionally have CPU
                    # data and GPU grads while active between block loads,
                    # so move the grad storage in place when the data is
                    # currently offloaded.
                    param.grad.data = moved.data

    def mark_compute_done(self, idx: int, event: torch.cuda.Event) -> None:
        assert self._pool is not None, "activate_pool not called"
        slot_id = self._block_to_slot.get(idx)
        if slot_id is not None:
            self._pool.set_compute_event(slot_id, event)

    def iter_trainables(
        self,
    ) -> "Iterator[tuple[PinnedParamBuffer, nn.Module, str]]":
        """Yield ``(buf, parent, leaf)`` for every trainable buffer
        across all blocks. Used by ``StreamedWeights.optimizer_step``
        to walk all trainables when bringing them to GPU around the
        optimizer-step boundary."""
        for block_bufs, block_locs in zip(
            self._param_bufs, self._param_locs, strict=True,
        ):
            for buf, (_qn, parent, leaf) in zip(block_bufs, block_locs, strict=True):
                if buf.requires_grad:
                    yield buf, parent, leaf

    def has_trainables(self) -> bool:
        """True if any block contains a trainable param. Used by the
        composer to decide whether to enforce checkpointing as a
        precondition for ``activate``."""
        return any(
            buf.requires_grad
            for block_bufs in self._param_bufs
            for buf in block_bufs
        )


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
    """Streams a single block list between pinned CPU and CUDA.

    The sharp, low-level streaming primitive. Manages the block list's
    owned params and buffers: pins them to CPU at construction time,
    streams them to CUDA via forward-pre hooks on :meth:`activate`.
    CPU activation is pass-through over that pinned host-backed state:
    no pool, no hooks, no copies. Frozen params use slot replacement;
    trainable params keep Parameter identity and swap only ``.data``.
    Does not touch parent modules, sibling modules, or out-of-block
    trainable parameters — those are the composer's responsibility.

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
    ``activate`` brings to CUDA or marks CPU active, ``deactivate`` returns slots to
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
    skip_slots:
        Optional set of :class:`SlotOwnership` tuples identifying
        ``(parent_module, leaf, kind)`` slots inside the blocks that
        the streamer should not pin / stream. Used by composers
        (typically :class:`ModelOffloader`) to surgically exclude
        slots that need a different lifecycle. The streamer manages
        the rest — both frozen params (slot replacement) and
        trainable params (``.data`` swap, identity-preserving) flow
        through the same slot pool.

        Trainable streaming requires activation checkpointing on
        every block (the ``.data`` swap bypasses autograd's
        version-counter check). The streamer doesn't enforce that
        precondition itself — :class:`ModelOffloader` does.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        *,
        blocks_to_swap: int,
        prefetch_count: int = 2,
        cyclic: bool = False,
        name: str | None = None,
        skip_slots: set[SlotOwnership] | None = None,
    ) -> None:
        self._blocks: list[nn.Module] = list(blocks)
        self._active_device: torch.device | None = None
        self._blocks_to_swap = blocks_to_swap
        self._prefetch_count = prefetch_count
        self._cyclic = cyclic
        self._name = name or f"StreamedWeights({len(self._blocks)} blocks)"

        if blocks_to_swap < 0:
            raise ValueError(f"blocks_to_swap ({blocks_to_swap}) must be >= 0")
        if blocks_to_swap >= len(self._blocks):
            raise ValueError(
                f"blocks_to_swap ({blocks_to_swap}) must be < num blocks ({len(self._blocks)})"
            )
        if prefetch_count < 0:
            raise ValueError(f"prefetch_count ({prefetch_count}) must be >= 0")

        # Pin in __init__ — uniform lifecycle with PinnedWeights, and
        # ModelCache integration sees a final `cache_bytes` immediately.
        # _BlockPinnedStore validates that every block shares the same
        # param-layout signature; mismatched configs raise here, before
        # any model slot is mutated. Heterogeneous block lists split
        # across separate `layers_attr=[...]` entries in ModelOffloader.
        for block in self._blocks:
            block.to("cpu")
        store = _BlockPinnedStore(
            self._blocks, skip_slots=skip_slots,
        )
        store.apply_slot_mutations()
        self._store: _BlockPinnedStore | None = store

        # Active resources allocated on CUDA activate().
        self._tracker: _BlockTracker | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._executor: ThreadPoolExecutor | None = None
        self._stream: torch.cuda.Stream | None = None
        self._pending: dict[int, Future[None]] = {}
        self._prefetch_events: dict[int, torch.cuda.Event] = {}
        self._last_idx: int = -1
        # Single-active-step invariant: nested optimizer_step()
        # would scatter outer state on top of inner updates, silently
        # discarding the outer optimizer step. Reentrant entry is
        # rejected at optimizer_step() entry.
        self._optimizer_step_active: bool = False

        # Auto-flush the CUDA allocator cache when the streamer is GC'd,
        # so callers don't need to remember an explicit empty_cache() at
        # workload boundaries. Captures only a bool (no self ref) so it
        # never blocks collection.
        weakref.finalize(
            self,
            _release_cuda_cache_on_drop,
            True,
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
    def param_aliases_per_block(
        self,
    ) -> list[list[list[tuple[str, nn.Module, str]]]]:
        """Per-block/per-buffer alias lists from the duplicate-aware walk.

        Parallel structure to :attr:`param_bufs_per_block`: each buffer has
        one or more ``(qualified_name, parent_module, leaf_name)`` aliases.
        """
        assert self._store is not None
        return self._store._param_aliases

    @property
    def cache_bytes(self) -> int:
        return self._store.cache_bytes if self._store is not None else 0

    @property
    def has_trainables(self) -> bool:
        return self._store is not None and self._store.has_trainables()

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        if device is not None:
            return canonical_device(device)
        raise ValueError(
            "StreamedWeights.activate() requires a device"
        )

    def _require_active_device(self) -> torch.device:
        device = self._active_device
        if device is None:
            raise RuntimeError("StreamedWeights has no active device")
        return device

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self, device: torch.device | str | None = None) -> None:
        """Activate the block list on ``device``.

        CUDA activation uses the streaming path: GPU slot pool, CUDA
        stream/events, prefetch executor, and forward hooks. CPU
        activation is pass-through over the pinned host-backed state.
        The composite's :meth:`activate` returns the model — this
        method returns ``None`` because the streamer doesn't own one.

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
        # Hard-guard against the documented "don't activate twice"
        # case. Without this, a double-activate would double-install
        # forward-pre hooks (silent grad doubling) and stack a second
        # slot pool on top of an active one.
        if self._active_device is not None:
            raise RuntimeError(
                "StreamedWeights.activate() called while already "
                "active. Deactivate first, or check for a leaked "
                "context manager."
            )
        active_device = self._resolve_device(device)
        if active_device.type == "cpu":
            self._activate_cpu_resolved()
            return
        if active_device.type != "cuda":
            raise ValueError(
                "StreamedWeights.activate() supports CUDA or CPU; "
                f"got {active_device}."
            )

        num_layers = len(self._blocks)
        num_resident = num_layers - self._blocks_to_swap
        num_gpu_slots = num_resident + self._prefetch_count

        self._active_device = active_device
        self._tracker = _BlockTracker()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._stream = torch.cuda.Stream(device=active_device, priority=-1)
        self._pending = {}
        self._prefetch_events = {i: torch.cuda.Event() for i in range(num_layers)}
        self._last_idx = -1

        self._store.activate_pool(num_gpu_slots, active_device)
        self._store.move_trainable_grads_to(active_device)

        for idx in range(min(num_resident, num_layers)):
            self._store.load_block(idx)
            self._tracker.mark_on_gpu(idx)

        self._register_hooks(num_resident)
        self.reset_peak()

        logger.info(
            f"{self._name} active: {self._blocks_to_swap}/{num_layers} on CPU, "
            f"{num_resident} resident on GPU, prefetch={self._prefetch_count}, "
            f"gpu_pool_slots={num_gpu_slots}"
        )

    def _activate_cpu_resolved(self) -> None:
        assert self._store is not None
        self._store.validate_cpu_activation_supported()
        self._active_device = torch.device("cpu")

    def deactivate(self) -> None:
        """Tear down active resources idempotently — safe to call
        before activate or multiple times. Every step in
        ``_teardown_active_resources`` null-checks; partial state
        from a failed activate cleans up correctly. Drop the
        strategy reference after deactivate to release pinned
        memory."""
        if self._active_device == torch.device("cpu"):
            self._active_device = None
            return
        prefetch_exc = self._teardown_active_resources()
        if prefetch_exc is not None:
            raise prefetch_exc

    @contextlib.contextmanager
    def use(self, device: torch.device | str) -> Iterator[None]:
        """Activate on ``device`` for the duration of the context."""
        self.activate(device)
        try:
            yield
        finally:
            self.deactivate()

    @contextlib.contextmanager
    def optimizer_step(self) -> Iterator[None]:
        """Context manager for the streamed-trainable optimizer boundary.

        On CUDA activation, this brings trainable ``.data`` to GPU
        around the optimizer-step boundary. On CPU activation, it is a
        guarded no-op because trainable ``.data`` is already resident in
        the pinned host-backed module slots.

        On CUDA, gradients live on GPU throughout backward via
        PyTorch's native ``AccumulateGrad`` — we don't D2H them inside
        this context. Only ``.data`` is materialized: streamed
        trainables are on pinned CPU after backward (the streamer's
        eviction path restores them) and the optimizer needs them on
        GPU.

        CUDA lifecycle:

        Enter:
          1. Quiesce streaming via :meth:`_drain_and_evict_all`
             (drain pending prefetch, sync prefetch stream, evict
             every allocated slot). Raises if a prefetch errored —
             eviction still ran first so the streamer is in a
             consistent baseline.
          2. On the streamer's private ``self._stream``, H2D each
             trainable's pinned ``.data`` to a fresh allocator-managed
             GPU buffer. Each H2D registers a rollback (repoint
             ``.data`` at pinned) on an :class:`~contextlib.ExitStack`.
          3. Have the user's current CUDA stream wait on
             ``self._stream`` so the optimizer (running on the user's
             stream) sees the materialized bytes.

        Exit (clean OR body exception):
          1. Have ``self._stream`` wait on the user's current stream
             so D2H sees the optimizer's writes.
          2. Blocking D2H of updated ``.data`` to the pinned clone
             on ``self._stream``, then sync. Blocking + sync so the
             next iteration's prefetch can't race against an
             unfinished D2H to the same pinned bytes.
          3. ExitStack unwinds: each materialized param's ``.data`` is
             repointed at its pinned clone, releasing the optimizer-step
             GPU allocation.

        CUDA failure modes:

        - **Enter raises mid-loop** (e.g., OOM on H2D for one
          trainable): ExitStack unwinds the rollbacks already
          registered, restoring ``.data`` to pinned for previously
          materialized params. No scatter (the data was never modified
          on GPU). Streamer is left in a clean post-quiesce state.
        - **Body raises after yield** (e.g., optimizer.step OOMs
          mid-iteration): scatter still runs, preserving whatever
          partial state the optimizer mutated, then the exception
          propagates. The pinned clones reflect the partial step;
          the user's exception handler sees the actual failure
          rather than a silent rollback to stale bytes.
        - **Reentrant entry**: rejected at top of the context
          manager. A nested optimizer-step boundary would discard the
          outer update.

        ``param.grad`` is untouched throughout: autograd manages it
        natively. ``optimizer.zero_grad()``, ``clip_grad_norm_``,
        ``GradScaler.unscale_``, and other grad-walking tools work
        orthogonally to the CUDA optimizer-step materialization window.

        Typical loop::

            loss.backward()
            with offloader.optimizer_step():
                optimizer.step()
            optimizer.zero_grad()  # can be inside or outside

        On CUDA, the slot pool's pre-warmed state is lost (next forward
        re-loads from pinned), but that cost is dominated by the
        forward + backward of the next iteration.
        """
        assert self._store is not None, (
            "StreamedWeights.optimizer_step() requires activate() to "
            "have been called and not yet deactivated."
        )
        if self._executor is None:
            if self._active_device is not None and self._active_device.type == "cpu":
                if self._optimizer_step_active:
                    raise RuntimeError(
                        "StreamedWeights.optimizer_step() does not support "
                        "reentrant entry."
                    )
                self._optimizer_step_active = True
                try:
                    yield
                finally:
                    self._optimizer_step_active = False
                return
            raise RuntimeError(
                "StreamedWeights.optimizer_step() called on inactive "
                "streamer. Use it inside the offloader's context "
                "manager, between backward and the next forward."
            )
        if self._optimizer_step_active:
            raise RuntimeError(
                "StreamedWeights.optimizer_step() does not support "
                "reentrant entry. A nested optimizer-step boundary would "
                "scatter the outer step's stale pinned bytes on top of "
                "the inner update."
            )
        if not self._store.has_trainables():
            yield
            return

        # Quiesce streaming via the unified drain+sync+evict path.
        # Raises if any pending prefetch errored — eviction still
        # runs first inside _drain_and_evict_all so the streamer
        # stays in a consistent state on the way out.
        first_prefetch_exc = self._drain_and_evict_all()
        if first_prefetch_exc is not None:
            raise first_prefetch_exc

        # Stream choreography: H2D runs on self._stream so it can
        # proceed without contention against work the user has on
        # the default stream. After H2D enqueueing, we have the
        # user's current_stream wait on self._stream before yielding,
        # so the optimizer (which runs on user's stream) sees the
        # materialized bytes. _drain_and_evict_all already synced
        # self._stream so it is safe to reuse here.
        target = self._require_active_device()
        step_stream = self._stream
        assert step_stream is not None, "stream allocated in activate()"

        self._optimizer_step_active = True
        try:
            with contextlib.ExitStack() as stack:
                # Each materialization registers a rollback that repoints
                # the user's Parameter's .data back at its pinned
                # cpu_param. ExitStack unwinds these on the way out
                # whether enter failed mid-loop (no scatter), the body
                # exited cleanly (scatter, then rollback), or the body
                # raised after yield (still scatter, then rollback) —
                # the rollback is idempotent w.r.t. a successful copy.
                materialized: list[tuple[nn.Parameter, PinnedParamBuffer]] = []
                with torch.cuda.stream(step_stream):
                    for buf, parent, leaf in self._store.iter_trainables():
                        param = _slot_param(parent, leaf)
                        param.data = buf.cpu_param.data.to(
                            target, non_blocking=True,
                        )
                        if param.grad is not None and param.grad.device != target:
                            param.grad = param.grad.to(target, non_blocking=True)
                        materialized.append((param, buf))
                        stack.callback(_repoint_data_to_pinned, param, buf)
                # User's current stream now waits for step_stream's
                # H2D. After this point the optimizer can safely read
                # param.data on its own stream.
                torch.cuda.current_stream(target).wait_stream(step_stream)

                try:
                    yield
                finally:
                    # Scatter on body-exit (clean OR exception). On body
                    # exception the optimizer may have mutated some params
                    # before raising; preserve that partial state rather
                    # than silently rolling back to pre-step pinned bytes.
                    # The blocking copy + sync guarantees the next prefetch
                    # reads stable bytes.
                    step_stream.wait_stream(torch.cuda.current_stream(target))
                    with torch.cuda.stream(step_stream):
                        for param, buf in materialized:
                            buf.cpu_param.data.copy_(
                                param.data, non_blocking=False,
                            )
                    step_stream.synchronize()
                    # ExitStack unwinds on `with` exit: each materialized
                    # param's .data is repointed at its pinned
                    # cpu_param.data, releasing the optimizer-step GPU
                    # allocation.
        finally:
            self._optimizer_step_active = False

    @contextlib.contextmanager
    def gather_for_step(self) -> Iterator[None]:
        """Backward-compatible alias for :meth:`optimizer_step`."""
        with self.optimizer_step():
            yield

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

    def _drain_and_evict_all(self) -> BaseException | None:
        """Quiesce streaming back to a baseline state.

        Drains pending prefetch futures (waits for completion),
        synchronizes the prefetch stream so any in-flight H2D has
        settled device-side, then evicts every block currently
        holding a pool slot via the store's source-of-truth
        (:meth:`_BlockPinnedStore.evict_allocated_blocks`), and
        clears tracker + pending bookkeeping.

        Idempotent — safe to call when no resources are active and
        safe to call multiple times. Returns the first prefetch
        exception encountered (or ``None``) so the caller can choose
        to surface it. We don't raise here so eviction still runs
        after a failed prefetch; without this, a transient prefetch
        error would leak the GPU slot it had partially populated.

        Used by both :meth:`_teardown_active_resources` (full
        deactivate) and :meth:`optimizer_step` (mid-cycle quiesce
        before optimizer step). Centralizing it removes the bug
        class where the two paths' bookkeeping diverged — optimizer_step
        used to walk ``_tracker._on_gpu`` and miss pending prefetch
        slots, leaking them across step boundaries.
        """
        first_prefetch_exc: BaseException | None = None
        for future in list(self._pending.values()):
            try:
                future.result()
            except BaseException as e:
                if first_prefetch_exc is None:
                    first_prefetch_exc = e
        self._pending.clear()

        if self._stream is not None:
            try:
                self._stream.synchronize()
            except BaseException as e:
                if first_prefetch_exc is None:
                    first_prefetch_exc = e

        if self._store is not None:
            self._store.evict_allocated_blocks()

        if self._tracker is not None:
            self._tracker.clear()

        return first_prefetch_exc

    def _teardown_active_resources(self) -> BaseException | None:
        """Idempotent cleanup of all active resources. Returns the
        first prefetch exception encountered (or None) so the caller
        can surface it after cleanup completes."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        # Shutdown executor first — shutdown(wait=True) waits for
        # already-submitted prefetches to finish. After that, no
        # background work can run, so the subsequent quiesce only
        # has to drain the futures' results (no in-flight risk).
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        first_prefetch_exc = self._drain_and_evict_all()

        self._prefetch_events.clear()

        if self._stream is not None:
            self._stream = None

        # Move trainable grads to CPU so deactivate's contract
        # (model on CPU after deactivate) holds for in-block
        # trainables. Backward leaves grads GPU-resident via
        # AccumulateGrad — this is the symmetric counterpart to
        # the ``.data`` restoration that already happened in
        # ``evict_allocated_blocks``.
        active_device = self._active_device
        if self._store is not None:
            self._store.move_trainable_grads_to(torch.device("cpu"))
            if active_device is None or active_device.type == "cuda":
                self._store.deactivate_pool()

        self._tracker = None
        self._last_idx = -1
        self._active_device = None

        return first_prefetch_exc

    def _evict_one(self, protected: set[int], compute_event: object | None = None) -> None:
        assert self._tracker is not None
        assert self._store is not None
        victim = self._tracker.pick_victim(protected=protected)
        if compute_event is not None:
            self._store.mark_compute_done(victim, cast(torch.cuda.Event, compute_event))
        self._store.release_slot(victim)
        self._tracker.mark_on_cpu(victim)

    def _do_prefetch(self, idx: int) -> None:
        assert self._stream is not None
        assert self._store is not None
        with torch.cuda.stream(self._stream):
            self._store.load_block(idx, non_blocking=True, stream=self._stream)
            self._prefetch_events[idx].record(self._stream)

    def _submit_prefetch(self, idx: int, max_on_gpu: int) -> None:
        assert self._tracker is not None
        assert self._executor is not None
        if idx < 0 or idx >= len(self._blocks):
            return
        if self._tracker.is_on_gpu(idx) or idx in self._pending:
            return
        if len(self._tracker._on_gpu) + len(self._pending) >= max_on_gpu:
            return
        self._pending[idx] = self._executor.submit(self._do_prefetch, idx)

    def _ensure_on_gpu(self, idx: int) -> None:
        assert self._tracker is not None
        assert self._store is not None
        future = self._pending.pop(idx, None)
        if future is not None:
            future.result()
            ev = self._prefetch_events[idx]
            if not ev.query():
                torch.cuda.current_stream(self._require_active_device()).wait_event(ev)
            self._tracker.mark_on_gpu(idx)
            return

        if not self._tracker.is_on_gpu(idx):
            self._store.load_block(idx)
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

        def _pre_hook(_module: nn.Module, _args: tuple[object, ...], *, idx: int) -> None:
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
            if tracker is None or streamer._store is None:
                return
            pending = streamer._pending
            if tracker.is_on_gpu(idx):
                tracker.touch(idx)
            else:
                compute_event = torch.cuda.current_stream(
                    streamer._require_active_device()
                ).record_event()
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
                direction = (
                    (-1 if diff > 0 else 1)
                    if cyclic and abs(diff) > wrap_threshold
                    else 1 if diff >= 0 else -1
                )
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
