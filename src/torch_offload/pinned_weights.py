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
  by letting :class:`PinnedParam` repoint plain ``Parameter.data``
  at pinned clones as each pinned parameter is created; if construction or
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
  single :class:`PinnedParam` and a single Parameter wrapper on
  activation, preserving the tying invariant on GPU.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch import nn

from ._devices import canonical_device
from .pinned_bindings import (
    PinnedBufferBinding,
    PinnedModuleBinding,
    PinnedModuleTarget,
    PinnedParamBinding,
    pin_module_slot_collection,
)
from .pinned_param import (
    PinnedParam,
    PostCopyHook,
    PostCopyHookHandle,
)
from .protocols import SlotKey
from .slot_collection import collect_module_slots
from .slots import (
    ParamSlot,
    assert_frozen,
)


def _validate_frozen_param_slot(slot: ParamSlot) -> None:
    assert_frozen(
        slot,
        owner="PinnedWeights",
        extra=(
            "Splitting a tied storage group between skip_slots "
            "and PinnedWeights silently breaks the alias on "
            "GPU — validate ties yourself if you go that route, "
            "or use ModelOffloader which handles it upstream."
        ),
    )


class PinnedWeights:
    """Whole-model pinned-CPU weight cache with bulk GPU transfer.

    Implements :class:`~torch_offload.protocols.ModelStrategy`.

    On construction, every frozen parameter slot is replaced with a
    Parameter wrapping pinned CPU storage (handling quanto decomposition
    and tied-weight dedup). :meth:`activate` allocates GPU tensors for
    each unique pinned parameter, swaps the matching Parameter into every
    slot that pointed at that pinned parameter, and returns the model;
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
        The model to cache. Managed slots may start on CPU or CUDA;
        construction clones them directly into pinned CPU storage.
    include_buffers:
        Also cache registered buffers (LayerNorm running stats, position
        embeddings stored as buffers, etc.). Default True. Set False
        for models with very large mutable buffers you'd rather rebuild
        on each call.
    skip_slots:
        Optional set of :class:`SlotKey` values identifying
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
        skip_slots: set[SlotKey] | None = None,
    ) -> None:
        self._model: nn.Module | None = model
        self._active_device: torch.device | None = None
        self._post_copy_hooks: dict[int, PostCopyHook] = {}
        skip: set[SlotKey] = skip_slots or set()

        # Phase 1: collect the model slots to manage without pinning or
        # slot mutation. This keeps skip/empty validation separate from
        # pinned-memory allocation.
        slot_collection = collect_module_slots(
            model,
            skip_slots=skip,
            include_buffers=include_buffers,
            param_group_by="storage",
            validate_param=_validate_frozen_param_slot,
        )
        # Reject before pinning if there is nothing at all to manage —
        # neither frozen params nor (when include_buffers=True)
        # registered buffers. Buffer-only modules (e.g., a pure
        # RoPE/positional table sibling) are valid: PinnedWeights still
        # gives them pinned-CPU storage and the activate/deactivate
        # round-trip, which is exactly what ModelOffloader non-block
        # composition needs.
        if (
            not slot_collection.param_slot_groups
            and not slot_collection.buffer_slot_groups
        ):
            raise ValueError(
                "PinnedWeights requires at least one frozen parameter or, "
                "when include_buffers=True, at least one registered buffer "
                "to cache. The wrapped model has neither — for training "
                "flows use torch_offload.ModelOffloader instead, or "
                "leave the model unwrapped."
            )

        # Phase 2: pin collected slot groups into bindings without
        # replacing module slots. PinnedParam intentionally repoints
        # plain Parameter.data at each pinned clone during this phase to
        # keep construction peak memory low. If a later pin fails, the
        # caller must drop the partially constructed model/strategy and
        # rebuild from a fresh model instance.
        self._binding: PinnedModuleBinding = pin_module_slot_collection(
            slot_collection,
        )

        # Phase 3: apply slot replacement/register_buffer mutations after
        # all pinning succeeded. This keeps module slot identity changes
        # grouped, but construction is not fully rollback-safe because of
        # the low-peak Parameter.data repointing described above.
        self._binding.place_on_pinned()

    # ------------------------------------------------------------------
    # ModelStrategy protocol
    # ------------------------------------------------------------------

    @property
    def param_bindings(self) -> list[PinnedParamBinding]:
        """Pinned parameter bindings managed by this instance.

        Used by :class:`~torch_offload.ModelOffloader` to build its
        target-name to pinned-param map.
        """
        return self._binding.param_bindings

    @property
    def buffer_bindings(self) -> list[PinnedBufferBinding]:
        """Pinned PyTorch buffer bindings managed by this instance."""
        return self._binding.buffer_bindings

    @property
    def cache_bytes(self) -> int:
        """Total pinned host bytes held. Tied weights counted once."""
        return self._binding.cache_bytes

    @property
    def model(self) -> nn.Module:
        """The wrapped model. Stable across activate/deactivate cycles."""
        assert self._model is not None
        return self._model

    @property
    def value(self) -> nn.Module:
        return self.model

    def register_post_copy_hook(
        self, pinned: PinnedParam, hook: PostCopyHook,
    ) -> PostCopyHookHandle:
        """Register a hook run after this component copies ``pinned`` to GPU.

        Package-internal: used by :class:`ModelOffloader` for merge-mode
        LoRA. Mirrors PyTorch's hook registration pattern by returning a
        handle whose :meth:`remove` method unregisters the hook.
        """
        if not self._binding.contains_pinned_param(pinned):
            raise ValueError(
                f"pinned param {pinned.name!r} is not owned by this PinnedWeights"
            )
        key = id(pinned)
        if key in self._post_copy_hooks:
            raise RuntimeError(
                "post-copy hook already registered for "
                f"pinned param {pinned.name!r}"
            )
        self._post_copy_hooks[key] = hook
        return PostCopyHookHandle(self._post_copy_hooks, key)

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
        if active_device.type == "cpu":
            self._binding.place_on_pinned()
        elif active_device.type == "cuda":
            # One active-device Parameter per unique pinned parameter.
            # Tied slots all receive the same Parameter object so the
            # tying invariant survives on device.
            target = PinnedModuleTarget(self._binding.pinned_params, active_device)
            self._binding.place_on_gpu(
                target,
                post_copy_hooks=self._post_copy_hooks,
                non_blocking=True,
            )
            torch.cuda.synchronize(active_device)
        else:
            raise ValueError(
                "PinnedWeights.activate() supports CUDA or CPU; "
                f"got {active_device}."
            )
        self._active_device = active_device

    def deactivate(self) -> None:
        """Repoint slots back at pinned-CPU Parameters. Idempotent —
        safe to call before activate or multiple times. After
        deactivate, drop the strategy reference to release pinned
        memory (and the model reference too if you don't need it
        anymore)."""
        try:
            self._binding.place_on_pinned()
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
