"""Whole-model pinned-CPU weight cache for fast bulk DMA to GPU.

Holds a model's frozen weights in pinned CPU memory so subsequent GPU
loads are bulk DMA (~200 ms for a 12 GB text encoder at PCIe Gen5 x16) instead
of re-reading the safetensors from disk (~3-5 s per call).

Use case: a model that fits on GPU when active but should be evicted
between calls — text encoder during diffusion, VAE between encode and
decode phases, etc. Different from :func:`ModelOffloader`: no per-block
streaming, no forward hooks, no LRU. The whole model goes to GPU on
:meth:`PinnedWeights.activate` and the GPU storage is released on
:meth:`PinnedWeights.deactivate` by repointing each module's parameter
slot back at a Parameter that wraps pinned CPU storage.

Implements :class:`~torch_offload.protocols.ModelStrategy` so it plugs
into a model cache directly.

Cross-cutting compatibility caveats (``torch.compile`` incompatibility,
DDP/FSDP wrap-before requirement, single-thread contract) live in the
:mod:`~torch_offload` package docstring.

Class-specific caveats
----------------------
- The constructor *mutates* the wrapped ``model`` — each frozen
  parameter slot (``module._parameters[leaf]``) is replaced with a
  Parameter wrapping pinned CPU storage, and registered buffers are
  replaced with pinned copies. Only use the model via :meth:`activate`
  or :meth:`use` after wrapping.
- Slot replacement (rather than ``param.data`` swap) is required for
  correctness with quanto ``WeightQBytesTensor``: assigning
  ``param.data = new_quanto_tensor`` is a no-op for the inner ``_data``
  / ``_scale`` storages, so the model would silently keep referencing
  the original (non-pinned) quanto wrapper.
- Buffer mutations during forward (RNN/SSM state, KV cache,
  training-mode BatchNorm running stats) are *discarded* on
  :meth:`deactivate`. Suitable for inference of stateless modules; not
  suitable for models that need persistent buffer state across calls.
- **Caller owns lifecycle correctness.** Calling :meth:`activate`
  twice without an intervening :meth:`deactivate` raises before slot
  movement or GPU allocation. Construction optimizes peak host memory
  by letting :class:`PinnedParamBuffer` repoint plain ``Parameter.data``
  at pinned clones as each buffer is created; if construction or
  activation raises after that point, retrying the same model/strategy
  is unsupported — drop references and rebuild from a fresh model
  instance.
- There is no ``close()``. Pinned memory is freed when the caller
  drops the strategy AND model references; Python's refcount-based
  GC reclaims the pinned tensors immediately. The strategy releases
  what it owns (its internal slot tracking); the user's model is the
  user's concern.
- Tied weights *are* deduplicated. Two parameter slots whose values
  share underlying storage — whether the standard ``tie_weights()``
  pattern (one ``Parameter`` under multiple names) or the rarer case
  of distinct quanto wrappers around shared inner ``_data`` — share a
  single :class:`PinnedParamBuffer` and a single Parameter wrapper on
  activation, preserving the tying invariant on GPU.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from ._devices import canonical_device
from .pinned_buffer import PinnedParamBuffer, storage_key
from .protocols import SlotOwnership
from .slots import BufferSlot, ParamSlot, assert_frozen, iter_buffer_slots, iter_param_slots

ParamSlotGroup = tuple[PinnedParamBuffer, list[tuple[nn.Module, str]]]
ParamAliasGroup = tuple[PinnedParamBuffer, list[tuple[str, nn.Module, str]]]


class PinnedWeights:
    """Whole-model pinned-CPU weight cache with bulk GPU transfer.

    Implements :class:`~torch_offload.protocols.ModelStrategy`.

    On construction, every frozen parameter slot is replaced with a
    Parameter wrapping pinned CPU storage (handling quanto decomposition
    and tied-weight dedup). :meth:`activate` allocates GPU tensors for
    each unique pinned buffer, swaps the matching Parameter into every
    slot that pointed at that buffer, and returns the model;
    :meth:`deactivate` swaps the slots back at the pinned-CPU
    Parameters so the GPU storage is released by refcount.

    Frozen-only by mechanism. Slot replacement installs fresh
    ``requires_grad=False`` Parameter wrappers, which orphans any
    optimizer state keyed by the user's pre-wrap Parameter — a
    trainable slot here would silently break training. Trainable
    slots must be partitioned out via ``skip_slots`` (the composer
    routes them to a separate strategy automatically; direct users
    are on the hook). An unskipped trainable slot raises at
    construction.

    Buffer-only modules (only registered buffers, no frozen params)
    are valid — common for sibling tables like RoPE/positional
    embeddings managed via :func:`ModelOffloader`'s non-block
    composition. Construction raises only if there is *nothing* to
    manage — neither frozen params nor (with ``include_buffers=True``)
    registered buffers.

    Parameters
    ----------
    model:
        The model to cache. Auto-moved to CPU at construction so
        ``pin_memory()`` succeeds.
    include_buffers:
        Also cache registered buffers (LayerNorm running stats, position
        embeddings stored as buffers, etc.). Default True. Set False
        for models with very large mutable buffers you'd rather rebuild
        on each call.
    skip_slots:
        Optional set of :class:`SlotOwnership` tuples identifying
        ``(parent_module, leaf, kind)`` slots to skip during the
        walk. Used by composers like :class:`ModelOffloader`
        that want to hand the *outer* model to PinnedWeights but
        manage some subset of slots themselves (block-streamed slots,
        trainable slots routed to a separate mover). Skipped slots are
        not pinned. Slot identity is based on
        ``(id(parent), leaf, kind)`` rather than ``id(param)`` so the
        filter survives slot-mutating strategies regardless of
        construction order.
    """

    def __init__(
        self,
        model: nn.Module,
        include_buffers: bool = True,
        *,
        skip_slots: set[SlotOwnership] | None = None,
    ) -> None:
        self._model: nn.Module | None = model
        self._include_buffers = include_buffers
        self._skip_slots: set[SlotOwnership] = skip_slots or set()
        self._active_device: torch.device | None = None

        # Auto-move to CPU so pin_memory() succeeds. Matches the
        # behavior of ModelOffloader — caller doesn't need to
        # remember the build-time device dance.
        model.to("cpu")

        # Phase 1: build all pinned templates without replacing module
        # slots. PinnedParamBuffer intentionally repoints plain
        # Parameter.data at each pinned clone during this phase to keep
        # construction peak memory low. If a later pin fails, the caller
        # must drop the partially constructed model/strategy and rebuild
        # from a fresh model instance.
        self._slots, self._param_aliases = self._collect_param_slots(model)
        self._buffer_slots = (
            self._collect_buffer_slots(model) if include_buffers else []
        )

        # Phase 2: apply slot replacement/register_buffer mutations after
        # all pinning succeeded. This keeps module slot identity changes
        # grouped, but construction is not fully rollback-safe because of
        # the low-peak Parameter.data repointing described above.
        for buf, locs in self._slots:
            for parent, leaf in locs:
                parent._parameters[leaf] = buf.cpu_param
        for pinned, locs in self._buffer_slots:
            for parent, leaf, persistent in locs:
                parent.register_buffer(leaf, pinned, persistent=persistent)

        # Reject only if there is nothing at all to manage — neither
        # frozen params nor (when include_buffers=True) registered
        # buffers. Buffer-only modules (e.g., a pure RoPE/positional
        # table sibling) are valid: PinnedWeights still gives them
        # pinned-CPU storage and the activate/deactivate round-trip,
        # which is exactly what ModelOffloader non-block composition
        # needs.
        if not self._slots and not self._buffer_slots:
            raise ValueError(
                "PinnedWeights requires at least one frozen parameter or, "
                "when include_buffers=True, at least one registered buffer "
                "to cache. The wrapped model has neither — for training "
                "flows use torch_offload.ModelOffloader instead, or "
                "leave the model unwrapped."
            )

    def _collect_param_slots(
        self, model: nn.Module
    ) -> tuple[list[ParamSlotGroup], list[ParamAliasGroup]]:
        # Tied-weight aware pinning. We walk with remove_duplicate=False so
        # shared submodule aliases and standard tie_weights() aliases both
        # show up, then group by storage identity.
        groups: dict[tuple[Any, ...], list[tuple[str, nn.Parameter, nn.Module, str]]] = {}
        for s in iter_param_slots(model):
            if s.slot in self._skip_slots:
                continue
            assert_frozen(
                s, owner="PinnedWeights",
                extra=(
                    "Splitting a tied storage group between skip_slots "
                    "and PinnedWeights silently breaks the alias on "
                    "GPU — validate ties yourself if you go that route, "
                    "or use ModelOffloader which handles it upstream."
                ),
            )
            groups.setdefault(self._param_storage_key(s), []).append(
                (s.name, s.param, s.parent, s.leaf)
            )

        slots: list[ParamSlotGroup] = []
        aliases: list[ParamAliasGroup] = []
        for members in groups.values():
            first_name, first_p = members[0][0], members[0][1]
            buf = PinnedParamBuffer(first_name, first_p)
            seen_locs: set[tuple[int, str]] = set()
            seen_aliases: set[str] = set()
            locs: list[tuple[nn.Module, str]] = []
            alias_locs: list[tuple[str, nn.Module, str]] = []
            for name, _, parent, leaf in members:
                if name not in seen_aliases:
                    seen_aliases.add(name)
                    alias_locs.append((name, parent, leaf))
                key = (id(parent), leaf)
                if key in seen_locs:
                    continue
                seen_locs.add(key)
                locs.append((parent, leaf))
            slots.append((buf, locs))
            aliases.append((buf, alias_locs))
        return slots, aliases

    def _collect_buffer_slots(
        self, model: nn.Module
    ) -> list[tuple[torch.Tensor, list[tuple[nn.Module, str, bool]]]]:
        buf_groups: dict[
            tuple[Any, ...],
            tuple[torch.Tensor, list[tuple[nn.Module, str, bool]]],
        ] = {}
        for s in iter_buffer_slots(model):
            if s.slot in self._skip_slots:
                continue
            persistent = s.leaf not in s.parent._non_persistent_buffers_set
            skey = self._buffer_storage_key(s)
            existing = buf_groups.get(skey)
            if existing is None:
                pinned = s.buffer.detach().clone(memory_format=torch.contiguous_format).pin_memory()
                buf_groups[skey] = (pinned, [(s.parent, s.leaf, persistent)])
            else:
                seen_locs = {(id(p), leaf) for p, leaf, _ in existing[1]}
                if (id(s.parent), s.leaf) not in seen_locs:
                    existing[1].append((s.parent, s.leaf, persistent))
        return list(buf_groups.values())

    @staticmethod
    def _param_storage_key(s: ParamSlot) -> tuple[Any, ...]:
        if s.param.numel() == 0:
            # Zero-sized tensors all share data_ptr()==0; key by object
            # identity so aliases of the same Parameter still dedupe.
            return ("__empty__", id(s.param))
        return storage_key(s.param.data)

    @staticmethod
    def _buffer_storage_key(s: BufferSlot) -> tuple[Any, ...]:
        if s.buffer.numel() == 0:
            return ("__empty_buf__", id(s.buffer))
        return storage_key(s.buffer)

    # ------------------------------------------------------------------
    # ModelStrategy protocol
    # ------------------------------------------------------------------

    @property
    def slots(self) -> list[ParamSlotGroup]:
        """Per-parameter ``(buffer, locations)`` pairs managed by this instance.

        Used by :class:`~torch_offload.ModelOffloader` to build a
        reverse index from parameter qualified names to their buffers.
        """
        return self._slots

    @property
    def param_aliases(self) -> list[ParamAliasGroup]:
        """Per-parameter ``(buffer, aliases)`` pairs managed by this instance.

        Each alias is ``(qualified_name, parent_module, leaf_name)`` from the
        duplicate-aware parameter walk. Tied weights share one buffer but keep
        every public name so LoRA target matching can resolve any alias.
        """
        return self._param_aliases

    @property
    def cache_bytes(self) -> int:
        """Total pinned host bytes held. Tied weights counted once."""
        total = 0
        for buf, _ in self._slots:
            total += buf.cache_bytes
        for pinned, _ in self._buffer_slots:
            total += pinned.numel() * pinned.element_size()
        return total

    @property
    def model(self) -> nn.Module:
        """The wrapped model. Stable across activate/deactivate cycles."""
        assert self._model is not None
        return self._model

    @property
    def value(self) -> nn.Module:
        return self.model

    def activate(self, device: torch.device | str | None = None) -> None:
        """Activate the wrapped model on ``device``.

        CUDA activation bulk-DMAs pinned weights to GPU: per-tensor
        ``.to()`` (non-blocking), then a single ``cuda.synchronize`` to
        make the writes visible. Tied parameter slots all receive the
        same GPU Parameter. CPU activation repoints slots back to the
        pinned CPU Parameters and performs no device copy. Reach the
        wrapped model via :attr:`model` once activated.

        Calling activate() twice without an intervening deactivate()
        raises before any slot movement or GPU allocation.

        **Activation failure semantics:** if CUDA activation fails
        midway, the strategy is left in an undefined partial state —
        some slots may be GPU, some pinned-CPU. Retrying activation on
        that strategy is unsupported; the caller's only supported
        cleanup path is :meth:`deactivate` (which forces all slots back
        to pinned-CPU) followed by dropping the strategy reference.
        """
        assert self._model is not None
        if self._active_device is not None:
            raise RuntimeError(
                "PinnedWeights.activate() called while already active "
                f"on {self._active_device}. Deactivate first, or check "
                "for a leaked context manager."
            )
        active_device = self._resolve_device(device)
        self._move_to_device(active_device)
        self._active_device = active_device

    def deactivate(self) -> None:
        """Repoint slots back at pinned-CPU Parameters. Idempotent —
        safe to call before activate or multiple times. After
        deactivate, drop the strategy reference to release pinned
        memory (and the model reference too if you don't need it
        anymore)."""
        try:
            self._move_to_pinned()
        finally:
            self._active_device = None

    @contextlib.contextmanager
    def use(self, device: torch.device | str) -> Iterator[nn.Module]:
        """Activate on ``device`` for the duration of the context."""
        self.activate(device)
        try:
            yield self.model
        finally:
            self.deactivate()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        if device is not None:
            return canonical_device(device)
        raise ValueError(
            "PinnedWeights.activate() requires a device; pass "
            "activate(device) or use this strategy through "
            "ModelCache.use(..., device=...)"
        )

    def _move_to_device(self, device: torch.device) -> None:
        if device.type == "cpu":
            if any(buf.transform is not None for buf, _ in self._slots):
                raise ValueError(
                    "PinnedWeights transforms require CUDA activation; "
                    f"got {device}."
                )
            self._move_to_pinned()
            return
        if device.type != "cuda":
            raise ValueError(
                "PinnedWeights.activate() supports CUDA or CPU; "
                f"got {device}."
            )

        # One active-device Parameter per unique buffer. Tied slots
        # all receive the same Parameter object so the tying invariant
        # survives on device.
        for buf, locs in self._slots:
            gpu_param = buf.load_to_gpu(device, non_blocking=True)
            for parent, leaf in locs:
                parent._parameters[leaf] = gpu_param
        if self._include_buffers:
            for pinned, locs in self._buffer_slots:
                gpu = pinned.to(device, non_blocking=True)
                for parent, leaf, persistent in locs:
                    parent.register_buffer(leaf, gpu, persistent=persistent)
        torch.cuda.synchronize(device)

    def _move_to_pinned(self) -> None:
        for buf, locs in self._slots:
            for parent, leaf in locs:
                parent._parameters[leaf] = buf.cpu_param
        if self._include_buffers:
            for pinned, locs in self._buffer_slots:
                for parent, leaf, persistent in locs:
                    parent.register_buffer(leaf, pinned, persistent=persistent)
