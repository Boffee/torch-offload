"""Pinned-CPU component for selected model parameters and buffers.

Holds selected slots in pinned CPU memory and bulk-copies them to the
activation device. This is the component used by :class:`ModelOffloader`
for both non-streamed slots and whole-model pinning. It satisfies
:class:`~torch_offload.protocols.ModelStrategyComponent`, not the
top-level model strategy protocol.

Cross-cutting compatibility caveats (``torch.compile`` incompatibility,
DDP/FSDP wrap-before requirement, single-thread contract) live in the
:mod:`~torch_offload` package docstring.

Class-specific caveats
----------------------
- The constructor *mutates* the wrapped ``model`` — frozen parameter
  slots (``module._parameters[leaf]``) are replaced with Parameters
  wrapping pinned CPU storage, trainable parameter ``.data`` points at
  pinned CPU storage while preserving the user's Parameter objects, and
  registered buffers are replaced with pinned copies.
- Slot replacement (rather than ``param.data`` swap) is required for
  correctness with quanto ``WeightQBytesTensor``: assigning
  ``param.data = new_quanto_tensor`` is a no-op for the inner ``_data``
  / ``_scale`` storages, so the model would silently keep referencing
  the original (non-pinned) quanto wrapper.
- Buffer mutations during forward (RNN/SSM state, KV cache,
  training-mode BatchNorm running stats) are *discarded* on
  :meth:`deactivate`. Suitable for inference of stateless modules; not
  suitable for models that need persistent buffer state across calls.
- Trainable parameter updates on CUDA must run inside
  :meth:`optimizer_step`. Without that boundary, deactivation restores
  older pinned CPU bytes and discards active GPU updates.
- **Caller owns lifecycle correctness.** Calling :meth:`activate`
  twice without an intervening :meth:`deactivate` raises before slot
  movement or GPU allocation. Construction optimizes peak host memory
  by letting :class:`PinnedParam` repoint plain ``Parameter.data``
  at pinned clones as each pinned parameter is created; if construction or
  activation raises after that point, retrying the same model/component
  is unsupported — drop references and rebuild from a fresh model
  instance.
- There is no ``close()``. Pinned memory is freed when the caller
  drops the component AND model references; Python's refcount-based
  GC reclaims the pinned tensors immediately. The component releases
  what it owns (its internal slot tracking); the user's model is the
  user's concern.
- Tied weights *are* deduplicated. Two parameter slots whose values
  share underlying storage — whether the standard ``tie_weights()``
  pattern (one ``Parameter`` under multiple names) or the rarer case
  of distinct quanto wrappers around shared inner ``_data`` — share a
  single :class:`PinnedParam` and a single target storage on
  activation, preserving the tying invariant on GPU.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator

import torch
from torch import nn

from ._devices import canonical_device
from .pinned_module import (
    PinnedModuleInstance,
    PinnedModuleStore,
    PinnedModuleTarget,
    PostCopyHook,
    PostCopyHookHandle,
)


class PinnedComponent:
    """Pinned-CPU slot component with bulk device transfer.

    On construction, every managed parameter is backed by pinned CPU
    storage (handling quanto decomposition and tied-weight dedup).
    :meth:`activate` allocates GPU tensors for each unique pinned
    parameter and installs that active storage into the managed model
    slots. Frozen parameters use slot replacement; trainable parameters
    preserve the user's Parameter objects and swap only ``.data`` so
    optimizer state remains valid. :meth:`deactivate` restores pinned
    CPU storage so GPU storage is released by refcount.

    If trainable params are active on CUDA, run ``optimizer.step()``
    inside :meth:`optimizer_step` so updated GPU bytes are copied back
    into the pinned CPU cache before the next deactivate/reactivate
    cycle.

    Buffer-only modules (only registered buffers, no params)
    are valid — common for sibling tables like RoPE/positional
    embeddings managed via :class:`ModelOffloader`'s non-block
    composition. Construction raises only if there is *nothing* to
    manage — neither selected params nor selected registered buffers.

    Parameters
    ----------
    model:
        The model to cache. Managed slots may start on CPU or CUDA;
        construction clones them directly into pinned CPU storage.
    include_param_names:
        Optional PyTorch parameter names to cache. ``None`` caches all
        parameters. An empty set caches none.
    include_buffer_names:
        Optional PyTorch buffer names to cache. ``None`` caches all
        registered buffers. An empty set caches none.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        include_param_names: Iterable[str] | None = None,
        include_buffer_names: Iterable[str] | None = None,
    ) -> None:
        self._model: nn.Module | None = model
        self._active_device: torch.device | None = None
        self._active_target: PinnedModuleTarget | None = None
        self._optimizer_step_active: bool = False

        # Pin selected names without replacing module slots.
        # PinnedParam intentionally repoints plain Parameter.data at each
        # pinned clone during this phase to keep construction peak memory
        # low. If a later pin fails, the caller must drop the partially
        # constructed model/strategy and rebuild from a fresh model instance.
        self._store = PinnedModuleStore.from_module(
            model,
            include_param_names=include_param_names,
            include_buffer_names=include_buffer_names,
        )

        # Reject if there is nothing at all to manage — neither selected
        # params nor selected registered buffers.
        # Buffer-only modules (e.g., a pure RoPE/positional table sibling)
        # are valid: PinnedComponent still gives them pinned-CPU storage and
        # the activate/deactivate round-trip, which is exactly what
        # ModelOffloader non-block composition needs.
        if not self._store.params and not self._store.buffers:
            raise ValueError(
                "PinnedComponent requires at least one parameter or, "
                "at least one registered buffer to cache. The selected "
                "model names contain neither — leave the model unwrapped."
            )

        # Bind this concrete model instance to the store. This applies the
        # module slot/register_buffer mutations after all pinning succeeded.
        # Construction is still not fully rollback-safe because of the
        # low-peak Parameter.data repointing described above.
        self._instance = PinnedModuleInstance.from_store(self._store, model)
        self._param_names = frozenset(self._store.params)
        self._buffer_names = frozenset(self._store.buffers)
        self._has_trainables = any(
            pinned.requires_grad for pinned in self._store.params.values()
        )

    # ------------------------------------------------------------------
    # Component API
    # ------------------------------------------------------------------

    @property
    def param_names(self) -> frozenset[str]:
        """Pinned parameter names managed by this instance."""
        return self._param_names

    @property
    def buffer_names(self) -> frozenset[str]:
        """Pinned buffer names managed by this instance."""
        return self._buffer_names

    @property
    def cache_bytes(self) -> int:
        """Total pinned host bytes held. Tied weights counted once."""
        return self._store.cache_bytes

    def register_post_copy_hook(
        self, name: str, hook: PostCopyHook,
    ) -> PostCopyHookHandle:
        """Register a hook after this component copies ``name`` to GPU.

        Package-internal: used by :class:`ModelOffloader` for merge-mode
        LoRA. Mirrors PyTorch's hook registration pattern by returning a
        handle whose :meth:`remove` method unregisters the hook.
        """
        return self._instance.register_post_copy_hook(name, hook)

    def post_copy_hook_key(self, name: str) -> int:
        """Stable hook/dedup key for a managed parameter name."""
        return self._instance.post_copy_hook_key(name)

    def activate(self, device: torch.device | str | None = None) -> None:
        """Activate the managed slots on ``device``.

        CUDA activation bulk-DMAs pinned weights to GPU: per-tensor
        ``.to()`` (non-blocking), then a single ``cuda.synchronize`` to
        make the writes visible. Tied parameter slots all receive the
        same GPU Parameter. CPU activation repoints slots back to the
        pinned CPU Parameters and performs no device copy.

        Calling activate() twice without an intervening deactivate()
        raises before any slot movement or GPU allocation.

        **Activation failure semantics:** if CUDA activation fails
        midway, the component is left in an undefined partial state —
        some slots may be GPU, some pinned-CPU. Retrying activation on
        that component is unsupported; the caller's only supported
        cleanup path is :meth:`deactivate` (which forces all slots back
        to pinned-CPU) followed by dropping the component reference.
        """
        assert self._model is not None
        if self._active_device is not None:
            raise RuntimeError(
                "PinnedComponent.activate() called while already active "
                f"on {self._active_device}. Deactivate first, or check "
                "for a leaked context manager."
            )
        active_device = self._resolve_device(device)
        if active_device.type == "cpu":
            self._instance.restore_pinned()
        elif active_device.type == "cuda":
            # One active-device Parameter per unique pinned parameter.
            # Tied slots all receive the same Parameter object so the
            # tying invariant survives on device.
            target = self._instance.allocate_target(active_device)
            self._instance.load_to_target(
                target,
                run_post_copy_hooks=True,
                non_blocking=True,
            )
            torch.cuda.synchronize(active_device)
            self._active_target = target
        else:
            raise ValueError(
                "PinnedComponent.activate() supports CUDA or CPU; "
                f"got {active_device}."
            )
        self._active_device = active_device

    def deactivate(self) -> None:
        """Repoint slots back at pinned-CPU Parameters. Idempotent —
        safe to call before activate or multiple times. After
        deactivate, drop the component reference to release pinned
        memory (and the model reference too if you don't need it
        anymore)."""
        try:
            self._instance.restore_pinned()
        finally:
            self._active_target = None
            self._active_device = None

    @contextlib.contextmanager
    def optimizer_step(self) -> Iterator[None]:
        """Optimizer-step boundary for managed trainable parameters.

        On CUDA activation, the model's trainable ``.data`` points at
        active GPU target storage. Wrap ``optimizer.step()`` in this
        context so updated trainable bytes are copied back into pinned
        CPU storage before the model is deactivated or reactivated.

        On CPU activation, or when inactive, this is a guarded no-op
        because trainable data is already resident in pinned CPU storage.
        ``param.grad`` is untouched.
        """
        if self._optimizer_step_active:
            raise RuntimeError(
                "PinnedComponent.optimizer_step() does not support reentrant entry."
            )

        self._optimizer_step_active = True
        try:
            active_device = self._active_device
            target = self._active_target
            if (
                self._has_trainables
                and active_device is not None
                and active_device.type == "cuda"
            ):
                if target is None:
                    raise RuntimeError(
                        "PinnedComponent optimizer-step state is inconsistent: "
                        "CUDA active without an active target."
                    )
                try:
                    yield
                finally:
                    self._instance.copy_trainables_from_target(
                        target,
                        non_blocking=True,
                    )
                    torch.cuda.synchronize(active_device)
            else:
                yield
        finally:
            self._optimizer_step_active = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        if device is not None:
            return canonical_device(device)
        raise ValueError(
            "PinnedComponent.activate() requires a device; pass "
            "activate(device) from the owning strategy/component."
        )


__all__ = [
    "PinnedComponent",
]
