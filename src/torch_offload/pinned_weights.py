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
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from typing import TypeVar

import torch
from torch import nn

from ._devices import canonical_device
from .pinned_module import (
    PinnedModuleInstance,
    PinnedModuleStore,
    PinnedModuleTarget,
    PostCopyHook,
)

_NamedT = TypeVar("_NamedT")


class _PostCopyHookHandle:
    __slots__ = ("_hooks", "_key")

    def __init__(
        self, hooks: MutableMapping[int, PostCopyHook], key: int,
    ) -> None:
        self._hooks: MutableMapping[int, PostCopyHook] | None = hooks
        self._key = key

    def remove(self) -> None:
        hooks = self._hooks
        if hooks is None:
            return
        hooks.pop(self._key, None)
        self._hooks = None


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

    Frozen-only by strategy contract. The lower pinned-module
    primitives can represent trainable params, but this whole-model
    strategy has no optimizer-step copy-back boundary: a CUDA optimizer
    update would mutate active target storage, then :meth:`deactivate`
    would restore the older pinned CPU bytes. Trainable parameters must
    be excluded via ``include_param_names`` (the composer routes them to
    a separate strategy automatically; direct users are on the hook).
    An included trainable parameter raises at construction.

    Buffer-only modules (only registered buffers, no frozen params)
    are valid — common for sibling tables like RoPE/positional
    embeddings managed via :func:`ModelOffloader`'s non-block
    composition. Construction raises only if there is *nothing* to
    manage — neither selected frozen params nor selected registered buffers.

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
        self._post_copy_hooks: dict[int, PostCopyHook] = {}

        params = _named_parameters(model)
        buffers = _named_buffers(model)
        selected_param_names = _select_names(
            "param",
            params,
            include_param_names,
        )
        selected_buffer_names = _select_names(
            "buffer",
            buffers,
            include_buffer_names,
        )
        _validate_frozen_params(params, selected_param_names)

        # Reject before pinning if there is nothing at all to manage —
        # neither frozen params nor selected registered buffers.
        # Buffer-only modules (e.g., a pure RoPE/positional table sibling)
        # are valid: PinnedWeights still gives them pinned-CPU storage and
        # the activate/deactivate round-trip, which is exactly what
        # ModelOffloader non-block composition needs.
        if not selected_param_names and not selected_buffer_names:
            raise ValueError(
                "PinnedWeights requires at least one frozen parameter or, "
                "at least one registered buffer to cache. The selected "
                "model names contain neither — for training flows use "
                "torch_offload.ModelOffloader instead, or leave the model "
                "unwrapped."
            )

        # Phase 2: pin selected names without replacing module slots.
        # PinnedParam intentionally repoints
        # plain Parameter.data at each pinned clone during this phase to
        # keep construction peak memory low. If a later pin fails, the
        # caller must drop the partially constructed model/strategy and
        # rebuild from a fresh model instance.
        self._store = PinnedModuleStore.from_module(
            model,
            include_param_names=selected_param_names,
            include_buffer_names=selected_buffer_names,
        )

        # Phase 3: bind this concrete model instance to the store. This
        # applies the module slot/register_buffer mutations after all
        # pinning succeeded. Construction is still not fully rollback-safe
        # because of the low-peak Parameter.data repointing described above.
        self._instance = PinnedModuleInstance.from_store(self._store, model)
        self._param_names = frozenset(self._store.params)
        self._buffer_names = frozenset(self._store.buffers)

    # ------------------------------------------------------------------
    # ModelStrategy protocol
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

    @property
    def model(self) -> nn.Module:
        """The wrapped model. Stable across activate/deactivate cycles."""
        assert self._model is not None
        return self._model

    @property
    def value(self) -> nn.Module:
        return self.model

    def register_post_copy_hook(
        self, name: str, hook: PostCopyHook,
    ) -> _PostCopyHookHandle:
        """Register a hook after this component copies ``name`` to GPU.

        Package-internal: used by :class:`ModelOffloader` for merge-mode
        LoRA. Mirrors PyTorch's hook registration pattern by returning a
        handle whose :meth:`remove` method unregisters the hook.
        """
        if name not in self._store.params:
            raise ValueError(
                f"param name {name!r} is not owned by this PinnedWeights"
            )
        key = self.post_copy_hook_key(name)
        if key in self._post_copy_hooks:
            raise RuntimeError(
                "post-copy hook already registered for "
                f"param name {name!r}"
            )
        self._post_copy_hooks[key] = hook
        return _PostCopyHookHandle(self._post_copy_hooks, key)

    def post_copy_hook_key(self, name: str) -> int:
        """Stable hook/dedup key for a managed parameter name."""
        return id(self._store.params[name])

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
            self._instance.restore_pinned()
        elif active_device.type == "cuda":
            # One active-device Parameter per unique pinned parameter.
            # Tied slots all receive the same Parameter object so the
            # tying invariant survives on device.
            target = self._instance.allocate_target(active_device)
            self._instance.load_to_target(
                target,
                post_copy_hooks=self._post_copy_hooks,
                non_blocking=True,
            )
            torch.cuda.synchronize(active_device)
            self._active_target = target
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
            self._instance.restore_pinned()
        finally:
            self._active_target = None
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


def _named_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    return _unique_name_dict(
        module.named_parameters(remove_duplicate=False),
        kind="parameter",
    )


def _named_buffers(module: nn.Module) -> dict[str, torch.Tensor]:
    return _unique_name_dict(
        module.named_buffers(remove_duplicate=False),
        kind="buffer",
    )


def _unique_name_dict(
    items: Iterable[tuple[str, _NamedT]],
    *,
    kind: str,
) -> dict[str, _NamedT]:
    values: dict[str, _NamedT] = {}
    for name, value in items:
        if name in values:
            raise ValueError(f"Module yielded duplicate {kind} name {name!r}.")
        values[name] = value
    return values


def _select_names(
    kind: str,
    items: Mapping[str, object],
    names: Iterable[str] | None,
) -> set[str]:
    if names is None:
        return set(items)

    selected = set(names)
    missing = sorted(selected - set(items))
    if missing:
        raise ValueError(
            f"PinnedWeights cannot include unknown {kind} names: "
            f"{_format_names(missing)}."
        )
    return selected


def _validate_frozen_params(
    params: dict[str, nn.Parameter],
    names: set[str],
) -> None:
    for name, param in params.items():
        if name not in names or not param.requires_grad:
            continue
        raise ValueError(
            f"PinnedWeights cannot manage trainable param {name!r}: "
            "this strategy has no optimizer-step copy-back boundary, "
            "so CUDA updates could be lost on deactivate. Use ModelOffloader "
            "(which partitions trainables into TrainableWeights "
            "automatically), or exclude the name and route it to a "
            "separate trainable mover."
        )


def _format_names(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in names)
