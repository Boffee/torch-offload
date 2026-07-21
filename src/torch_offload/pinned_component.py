"""Pinned-CPU component for selected model parameters and buffers.

Holds selected parameters and buffers in pinned CPU memory and
bulk-copies them to the activation device. This is the component used
by :class:`ModelOffloader` for both non-streamed names and whole-model
pinning. It is a composable activate/deactivate lifecycle piece, not a
top-level model runtime.

Cross-cutting compatibility caveats (``torch.compile`` incompatibility,
DDP/FSDP wrap-before requirement, single-thread contract) live in the
:mod:`~torch_offload` package docstring.

Class-specific caveats
----------------------
- Binding mutates the wrapped ``model`` — frozen parameter
  registry entries (``module._parameters[leaf]``) are replaced with Parameters
  wrapping pinned CPU storage, trainable parameter ``.data`` points at
  pinned CPU storage while preserving the user's Parameter objects, and
  registered buffers are replaced with pinned copies.
- Registry replacement (rather than ``param.data`` swap) is required for
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
  twice without an intervening :meth:`deactivate` raises before registry
  movement or GPU allocation. Construction optimizes peak host memory
  by letting :class:`PinnedParam` repoint plain ``Parameter.data``
  at pinned clones as each pinned parameter is created; if construction or
  activation raises after that point, retrying the same model/component
  is unsupported — drop references and rebuild from a fresh model
  instance.
- There is no ``close()``. Pinned memory is freed when the caller
  drops the component AND model references; Python's refcount-based
  GC reclaims the pinned tensors immediately. The component releases
  what it owns (its internal name tracking); the user's model is the
  user's concern.
- Tied weights *are* deduplicated. Two parameter names whose values
  share underlying storage — whether the standard ``tie_weights()``
  pattern (one ``Parameter`` under multiple names) or the rarer case
  of distinct quanto wrappers around shared inner ``_data`` — share a
  single :class:`PinnedParam` and a single target storage on
  activation, preserving the tying invariant on GPU.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import torch
from torch import nn

from ._devices import canonical_device
from .pinned_module import (
    PinnedModuleInstance,
    PinnedModuleStore,
    PinnedModuleTarget,
    PostCopyHook,
)


@dataclass(frozen=True, slots=True)
class PinnedComponentStore:
    """Reusable pinned backing storage for :class:`PinnedComponent`.

    Component-level wrapper over the internal name-keyed module store.
    Build once from a prototype module, then bind it to concrete
    compatible modules with :meth:`bind`.
    """

    _module_store: PinnedModuleStore

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        include_param_names: Iterable[str] | None = None,
        include_buffer_names: Iterable[str] | None = None,
    ) -> PinnedComponentStore:
        """Pin selected model state into a reusable component store."""
        return cls(
            PinnedModuleStore.from_module(
                model,
                include_param_names=include_param_names,
                include_buffer_names=include_buffer_names,
            )
        )

    @property
    def param_names(self) -> frozenset[str]:
        """Pinned parameter names in this store."""
        return frozenset(self._module_store.params)

    @property
    def buffer_names(self) -> frozenset[str]:
        """Pinned buffer names in this store."""
        return frozenset(self._module_store.buffers)

    @property
    def cache_bytes(self) -> int:
        """Total pinned host bytes held by this store."""
        return self._module_store.cache_bytes

    @property
    def has_trainables(self) -> bool:
        """Whether any pinned parameter is trainable."""
        return self._module_store.has_trainables

    def bind(self, model: nn.Module) -> PinnedComponent:
        """Bind this store's pinned backing bytes to ``model``."""
        return PinnedComponent(self._module_store.bind(model))


class PinnedComponent:
    """Pinned-CPU component with bulk device transfer.

    Instances are created by binding a :class:`PinnedComponentStore` to a
    compatible model. Every managed parameter is backed by pinned CPU
    storage (handling quanto decomposition and tied-weight dedup).
    :meth:`activate` allocates GPU tensors for each unique pinned
    parameter and installs that active storage into the managed model
    registry entries. Frozen parameters use registry replacement;
    trainable parameters preserve the user's Parameter objects and swap
    only ``.data`` so optimizer state remains valid.
    :meth:`deactivate` restores pinned CPU storage so GPU storage is
    released by refcount.

    If trainable params are active on CUDA, run ``optimizer.step()``
    inside :meth:`optimizer_step` so updated GPU bytes are copied back
    into the pinned CPU cache before the next deactivate/reactivate
    cycle.

    Buffer-only modules (only registered buffers, no params)
    are valid — common for sibling tables like RoPE/positional
    embeddings managed via :class:`ModelOffloader`'s non-block
    composition. Empty selections are valid no-op components; the
    top-level :class:`ModelOffloader` still rejects configurations
    with no components to manage.

    Stores are constructed with :meth:`PinnedComponentStore.from_module`.
    Managed tensors may start on CPU or CUDA; store construction clones
    them directly into pinned CPU storage.
    """

    def __init__(self, instance: PinnedModuleInstance) -> None:
        if not isinstance(instance, PinnedModuleInstance):
            raise TypeError(
                "PinnedComponent requires a PinnedModuleInstance; "
                "use PinnedComponentStore.from_module(model).bind(model)."
            )
        self._instance = instance
        self._param_names = frozenset(instance.params)
        self._buffer_names = frozenset(instance.buffers)
        self._has_trainables = instance.has_trainables
        self._active_device: torch.device | None = None
        self._active_target: PinnedModuleTarget | None = None
        self._optimizer_step_active: bool = False

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

    def register_post_copy_hook(
        self, name: str, hook: PostCopyHook,
    ) -> Callable[[], None]:
        """Register a hook after this component copies ``name`` to GPU.

        Package-internal: used by :class:`ModelOffloader` for merge-mode
        LoRA. Returns a callable that unregisters the hook.
        """
        return self._instance.register_post_copy_hook(name, hook)

    def activate(self, device: torch.device, **kwargs: object) -> None:
        """Activate the managed tensors on ``device``.

        CUDA activation bulk-DMAs pinned weights to GPU: per-tensor
        ``.to()`` (non-blocking), then a single ``cuda.synchronize`` to
        make the writes visible, then realigns any retained trainable
        ``.grad`` to the GPU so the next backward accumulates on-device
        (mirrors :meth:`deactivate` moving grad to CPU). Tied parameter
        names all receive the same GPU Parameter. CPU activation repoints
        registry entries back to the pinned CPU Parameters and performs no
        device copy.

        Calling activate() twice without an intervening deactivate()
        raises before any registry movement or GPU allocation.

        **Activation failure semantics:** if CUDA activation fails
        midway, the component is left in an undefined partial state —
        some tensors may be GPU, some pinned-CPU. Retrying activation on
        that component is unsupported; the caller's only supported
        cleanup path is :meth:`deactivate` (which forces managed
        tensors back to pinned-CPU) followed by dropping the component
        reference.
        """
        del kwargs  # streaming-only policy; bulk-pinned activation ignores it
        if self._active_device is not None:
            raise RuntimeError(
                "PinnedComponent.activate() called while already active "
                f"on {self._active_device}. Deactivate first, or check "
                "for a leaked context manager."
            )
        active_device = canonical_device(device)
        if active_device.type == "cpu":
            self._instance.install_pinned()
        elif active_device.type == "cuda":
            # One active-device Parameter per unique pinned parameter.
            # Tied names all receive the same Parameter object so the
            # tying invariant survives on device.
            target = self._instance.allocate_target(active_device)
            self._instance.load_to_target(
                target,
                run_post_copy_hooks=True,
                non_blocking=True,
            )
            torch.cuda.synchronize(active_device)
            # Realign trainable grads with their now-GPU data so the next
            # backward accumulates on-device. A no-op unless a prior CPU
            # optimizer step left a retained CPU grad (set_to_none=False).
            self._instance.move_trainable_grads_to(active_device)
            self._active_target = target
        else:
            raise ValueError(
                "PinnedComponent.activate() supports CUDA or CPU; "
                f"got {active_device}."
            )
        self._active_device = active_device

    def deactivate(self) -> None:
        """Repoint registry entries back at pinned-CPU Parameters. Idempotent —
        safe to call before activate or multiple times. After
        deactivate, drop the component reference to release pinned
        memory (and the model reference too if you don't need it
        anymore).

        Trainable ``.grad`` follows ``.data`` to pinned CPU here (grads
        otherwise linger wherever ``AccumulateGrad`` left them, i.e. on the
        GPU, pinning device memory and stranding the gradient off-host).
        This gives a uniform deactivated resting state — ``.data`` and
        ``.grad`` both on pinned CPU — so a context-free CPU
        ``optimizer.step()`` works the same for pinned and streamed
        trainables."""
        try:
            self._instance.install_pinned()
            self._instance.move_trainable_grads_to(torch.device("cpu"))
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

        This context does not move ``param.grad``; grad placement is owned
        by the activate/deactivate cycle, which keeps grad on the same
        device as ``.data`` (GPU while active, pinned CPU once deactivated).
        Use this context to step on the GPU; to step on the CPU instead,
        call ``optimizer.step()`` while deactivated.
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

__all__ = [
    "PinnedComponent",
    "PinnedComponentStore",
]
