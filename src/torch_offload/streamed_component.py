"""Block-streaming primitive for memory-efficient training and inference.

A :class:`StreamedComponentStore` manages reusable pinned CPU backing
storage for a single block list whose blocks share the same parameter
layout (names, shapes, dtypes, and any tensor-adapter wrapper metadata).
Binding that store to a compatible model creates a
:class:`StreamedComponent` that streams the resolved blocks to GPU on
demand via forward-pre hooks, and uses a pre-allocated GPU target pool plus
a background prefetcher to overlap DMA with compute. On CPU, the
host-backed pinned state is used directly without streaming. Heterogeneous
block lists (e.g. Flux's two block kinds) split into multiple
:class:`StreamedComponent` instances composed via :class:`ModelOffloader`.

In-block trainable params (LoRA adapters) flow through the same target
pool; pinned module instances branch on the source trainable flag to swap
``.data`` (preserves user Parameter identity for autograd / optimizer
state) instead of replacing the Parameter wrapper. Gradients live on GPU
during backward via PyTorch's native ``AccumulateGrad``; only ``.data``
is materialized around ``optimizer.step()`` via :meth:`optimizer_step`.

This is the sharp, low-level primitive. It does NOT manage:

- Non-block parts of the model (parent-module state, sibling
  modules) — caller derives :class:`PinnedComponent` include-name sets
  by excluding the streamer's owned names.
- Out-of-block trainable parameter movement — caller handles that
  alongside non-streamed parameters, usually with :class:`PinnedComponent`.
- Shared storage with tensors outside the streamed block list — caller
  must choose a valid composition; use whole-model
  :class:`PinnedComponent` if sharing must be preserved.
- Activation-checkpointing enforcement — required for in-block
  trainable streaming, but checked at the composer level.

Most users want :class:`ModelOffloader` (the blessed safe API). Reach for
:class:`StreamedComponentStore` / :class:`StreamedComponent` directly only
when you need bespoke composition (e.g., multiple block lists like Flux's
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
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from ._devices import canonical_device
from .module_names import group_names, walk_attr_path
from .pinned_buffer import PinnedBuffer
from .pinned_module import (
    PinnedModuleInstance,
    PinnedModuleStore,
    PinnedModuleTarget,
    PostCopyHook,
    PostCopyHookHandle,
)
from .pinned_param import PinnedParam
from .tensor_adapter_registry import buffer_tensor_id, param_tensor_id

logger = logging.getLogger(__name__)

_LoadedTrainableBlock = tuple[PinnedModuleInstance, PinnedModuleTarget]


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
# Pre-allocated GPU target pool
# ---------------------------------------------------------------------------


class _PinnedModuleTargetPool:
    """Pool of pre-allocated :class:`PinnedModuleTarget` instances."""

    def __init__(
        self,
        template_instance: PinnedModuleInstance,
        num_targets: int,
        device: torch.device,
    ) -> None:
        self._targets = [
            template_instance.allocate_target(device)
            for _ in range(num_targets)
        ]
        self._free: list[int] = list(range(num_targets))
        self._events: list[torch.cuda.Event | None] = [None] * num_targets

    def acquire(self) -> int:
        return self._free.pop()

    def release(self, target_idx: int) -> None:
        self._free.append(target_idx)

    def target(self, target_idx: int) -> PinnedModuleTarget:
        return self._targets[target_idx]

    def set_compute_event(self, target_idx: int, event: torch.cuda.Event) -> None:
        self._events[target_idx] = event

    def wait_if_needed(
        self, target_idx: int, stream: torch.cuda.Stream | None
    ) -> None:
        ev = self._events[target_idx]
        if ev is not None:
            if stream is not None and not ev.query():
                stream.wait_event(ev)
            self._events[target_idx] = None


# ---------------------------------------------------------------------------
# Streamed block instances
# ---------------------------------------------------------------------------


def _param_target_layout(p: nn.Parameter) -> tuple[object, object]:
    """Layout fields that block 0's pool template must match across blocks.

    ``Tensor.copy_`` silently casts dtype and silently broadcasts
    compatible shapes — both invisible failure modes that would
    silently corrupt a load. Wrapper metadata (qtype, axis,
    activation_qtype, quant_type) is similarly invisible to copy_.

    :class:`PinnedParam` owns the tensor-adapter details needed for the
    opaque target layout because wrapper metadata is type-specific. The
    returned value intentionally excludes tensor identity so distinct
    block instances with the same layout can share one pool template.
    """
    return PinnedParam.target_layout_for(p)


def _buffer_target_layout(buffer: torch.Tensor) -> tuple[object, ...]:
    return PinnedBuffer.target_layout_for(buffer)


def _check_block_layouts_match(
    block_params: Sequence[dict[str, nn.Parameter]],
    block_buffers: Sequence[dict[str, torch.Tensor]],
) -> None:
    """Raise if blocks have mismatched param/buffer layouts. Called before
    pinning so layout failures leave parameters, buffers, and storage untouched.

    See :func:`_param_target_layout` for what counts as "matched."
    """
    if len(block_params) <= 1:
        return

    def sig(params: dict[str, nn.Parameter]) -> tuple:
        return tuple(
            (
                tuple(names),
                params[names[0]].requires_grad,
                _param_target_layout(params[names[0]]),
            )
            for names in group_names(
                params.keys(),
                lambda name: param_tensor_id(params[name]),
            )
        )

    def buffer_sig(buffers: dict[str, torch.Tensor]) -> tuple:
        return tuple(
            (
                tuple(names),
                _buffer_target_layout(buffers[names[0]]),
            )
            for names in group_names(
                buffers.keys(),
                lambda name: buffer_tensor_id(buffers[name]),
            )
        )

    ref = sig(block_params[0])
    ref_buffers = buffer_sig(block_buffers[0])
    for i in range(1, len(block_params)):
        if sig(block_params[i]) != ref:
            raise ValueError(
                f"Block {i} param layout differs from block 0. "
                "All blocks in a StreamedComponent group must share the "
                "same param structure (names, storage grouping, "
                "requires_grad, shapes, dtypes, and any tensor-adapter "
                "wrapper metadata). Split heterogeneous block lists "
                "across separate `blocks_attr=[...]` groups in "
                "ModelOffloader."
            )
        if buffer_sig(block_buffers[i]) != ref_buffers:
            raise ValueError(
                f"Block {i} buffer layout differs from block 0. "
                "All blocks in a StreamedComponent group must share the "
                "same buffer structure (names, shapes, dtypes, and "
                "tensor layouts). Split heterogeneous block lists "
                "across separate `blocks_attr=[...]` groups in "
                "ModelOffloader."
            )


def _collect_streamed_entries(
    blocks: list[nn.Module],
    stream_param_names: set[str] | None,
    stream_buffer_names: set[str] | None,
) -> tuple[list[dict[str, nn.Parameter]], list[dict[str, torch.Tensor]]]:
    block_params: list[dict[str, nn.Parameter]] = []
    block_buffers: list[dict[str, torch.Tensor]] = []

    for block in blocks:
        params, buffers = _select_streamed_entries(
            block,
            stream_param_names,
            stream_buffer_names,
        )
        block_params.append(params)
        block_buffers.append(buffers)

    return block_params, block_buffers


def _select_streamed_entries(
    block: nn.Module,
    stream_param_names: set[str] | None,
    stream_buffer_names: set[str] | None,
) -> tuple[dict[str, nn.Parameter], dict[str, torch.Tensor]]:
    params: dict[str, nn.Parameter] = {}
    all_param_names: set[str] = set()
    for name, param in block.named_parameters(remove_duplicate=False):
        all_param_names.add(name)
        if stream_param_names is not None and name not in stream_param_names:
            continue
        params[name] = param
    _validate_streamed_names_known(stream_param_names, all_param_names)

    buffers: dict[str, torch.Tensor] = {}
    all_buffer_names: set[str] = set()
    for name, buffer in block.named_buffers(remove_duplicate=False):
        all_buffer_names.add(name)
        if stream_buffer_names is not None and name not in stream_buffer_names:
            continue
        buffers[name] = buffer
    _validate_streamed_names_known(stream_buffer_names, all_buffer_names)

    return params, buffers


def _validate_streamed_names_known(
    names: set[str] | None,
    known_names: set[str],
) -> None:
    if names is None:
        return
    missing = sorted(names - known_names)
    if missing:
        raise ValueError(
            "StreamedComponent cannot select unknown block-local names: "
            f"{_format_names(missing)}."
        )


def _format_names(names: Sequence[str]) -> str:
    return ", ".join(repr(name) for name in names)


def _pin_block_module_stores(
    blocks: Sequence[nn.Module],
    *,
    stream_param_names: set[str] | None = None,
    stream_buffer_names: set[str] | None = None,
) -> list[PinnedModuleStore]:
    """Collect, validate, and pin one :class:`PinnedModuleStore` per block.

    Pre-pin validation failures do not pin and do not mutate module
    parameters or buffers. Once pinning starts, :class:`PinnedParam` may use its
    low-peak ``Parameter.data`` repointing optimization; recovery from
    a pin-time failure is unsupported, matching :class:`PinnedComponent`.
    """
    # Walk each block to collect selected param/buffer names WITHOUT
    # pinning anything. Pinning runs only after the block layout check
    # passes, so invalid configurations raise before model mutation.
    block_params, block_buffers = _collect_streamed_entries(
        list(blocks),
        stream_param_names,
        stream_buffer_names,
    )

    # Validate before pinning. ``Tensor.copy_`` silently casts dtype and
    # silently broadcasts compatible shapes, so any block N with
    # mismatched dtype, name, or wrapper metadata would otherwise load
    # into block 0's pool target without raising and corrupt forward.
    _check_block_layouts_match(block_params, block_buffers)

    return [
        PinnedModuleStore.from_module(
            block,
            include_param_names=params,
            include_buffer_names=buffers,
        )
        for block, params, buffers in zip(
            blocks, block_params, block_buffers, strict=True,
        )
    ]


def _iter_instance_trainable_params(
    instance: PinnedModuleInstance,
) -> Iterator[nn.Parameter]:
    params = dict(instance.module.named_parameters(remove_duplicate=False))
    seen: set[int] = set()
    for name, pinned in instance.params.items():
        if not pinned.requires_grad:
            continue
        param = params[name]
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        yield param


def _move_instance_trainable_grads_to(
    instance: PinnedModuleInstance,
    device: torch.device,
) -> None:
    for param in _iter_instance_trainable_params(instance):
        if param.grad is None or param.grad.device == device:
            continue
        moved = param.grad.to(device)
        if param.data.device == device:
            param.grad = moved
        else:
            # PyTorch's grad setter rejects cross-device grad/data
            # pairs. Streamed trainables intentionally have CPU data
            # and GPU grads while active between block loads, so move
            # the grad storage in place when the data is currently
            # offloaded.
            param.grad.data = moved.data


def _build_param_name_index(
    instances: Sequence[PinnedModuleInstance],
    prefix: str | None,
) -> dict[str, tuple[int, str]]:
    index: dict[str, tuple[int, str]] = {}
    for block_idx, instance in enumerate(instances):
        for local_name in instance.params:
            name = _streamed_param_name(prefix, block_idx, local_name)
            if name in index:
                raise ValueError(
                    f"duplicate streamed parameter name {name!r}"
                )
            index[name] = (block_idx, local_name)
    return index


def _build_buffer_name_index(
    instances: Sequence[PinnedModuleInstance],
    prefix: str | None,
) -> dict[str, tuple[int, str]]:
    index: dict[str, tuple[int, str]] = {}
    for block_idx, instance in enumerate(instances):
        for local_name in instance.buffers:
            name = _streamed_param_name(prefix, block_idx, local_name)
            if name in index:
                raise ValueError(
                    f"duplicate streamed buffer name {name!r}"
                )
            index[name] = (block_idx, local_name)
    return index


def _streamed_param_name(
    prefix: str | None,
    block_idx: int,
    local_name: str,
) -> str:
    name = f"{block_idx}.{local_name}"
    return name if prefix is None else f"{prefix}.{name}"


def _streamed_log_label(name: str | None, block_count: int) -> str:
    if name is None:
        return f"StreamedComponent({block_count} blocks)"
    return f"StreamedComponent[{name}]"


def _validate_stream_config(
    *,
    num_blocks: int,
    blocks_to_swap: int,
    prefetch_count: int,
) -> None:
    if blocks_to_swap < 0:
        raise ValueError(f"blocks_to_swap ({blocks_to_swap}) must be >= 0")
    if blocks_to_swap >= num_blocks:
        raise ValueError(
            f"blocks_to_swap ({blocks_to_swap}) must be < num blocks ({num_blocks})"
        )
    if prefetch_count < 0:
        raise ValueError(f"prefetch_count ({prefetch_count}) must be >= 0")


def _resolve_blocks(module: nn.Module, blocks_path: str) -> list[nn.Module]:
    obj = walk_attr_path(module, blocks_path)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(
            f"Expected nn.ModuleList at '{blocks_path}', got {type(obj).__name__}"
        )
    blocks = list(obj)
    if not blocks:
        raise ValueError(f"blocks_attr = {blocks_path!r} resolved to empty list")
    return blocks


def _streamed_param_names_for_blocks(
    blocks: Sequence[nn.Module],
    *,
    stream_trainables: bool,
) -> set[str]:
    param_names = _block_param_names(blocks[0], stream_trainables=stream_trainables)
    for i, block in enumerate(blocks[1:], start=1):
        if _block_param_names(block, stream_trainables=stream_trainables) != param_names:
            raise ValueError(
                f"Block {i} param layout differs from block 0: selected "
                "parameter names differ. All blocks in a StreamedComponent "
                "group must select the same parameter names."
            )
    return param_names


def _streamed_buffer_names_for_blocks(blocks: Sequence[nn.Module]) -> set[str]:
    buffer_names = _block_buffer_names(blocks[0])
    for i, block in enumerate(blocks[1:], start=1):
        if _block_buffer_names(block) != buffer_names:
            raise ValueError(
                f"Block {i} buffer layout differs from block 0: selected "
                "buffer names differ. All blocks in a StreamedComponent "
                "group must select the same buffer names."
            )
    return buffer_names


def _block_param_names(
    block: nn.Module,
    *,
    stream_trainables: bool,
) -> set[str]:
    return {
        name
        for name, param in block.named_parameters(remove_duplicate=False)
        if stream_trainables or not param.requires_grad
    }


def _block_buffer_names(block: nn.Module) -> set[str]:
    return {
        name
        for name, _buffer in block.named_buffers(remove_duplicate=False)
    }


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


@dataclass(frozen=True, slots=True)
class StreamedComponentStore:
    """Reusable pinned backing storage for a streamed block group."""

    _block_stores: tuple[PinnedModuleStore, ...]
    blocks_path: str
    blocks_to_swap: int
    prefetch_count: int = 2
    cyclic: bool = False

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        blocks_path: str,
        blocks_to_swap: int,
        prefetch_count: int = 2,
        cyclic: bool = False,
        stream_trainable_weights: bool = False,
    ) -> StreamedComponentStore:
        """Resolve ``blocks_path`` on ``model`` and pin its streamed blocks."""
        blocks = _resolve_blocks(model, blocks_path)
        _validate_stream_config(
            num_blocks=len(blocks),
            blocks_to_swap=blocks_to_swap,
            prefetch_count=prefetch_count,
        )
        stream_param_names = _streamed_param_names_for_blocks(
            blocks,
            stream_trainables=stream_trainable_weights,
        )
        stream_buffer_names = _streamed_buffer_names_for_blocks(blocks)
        block_stores = _pin_block_module_stores(
            blocks,
            stream_param_names=stream_param_names,
            stream_buffer_names=stream_buffer_names,
        )
        return cls(
            _block_stores=tuple(block_stores),
            blocks_path=blocks_path,
            blocks_to_swap=blocks_to_swap,
            prefetch_count=prefetch_count,
            cyclic=cyclic,
        )

    @property
    def streamed_param_names_by_block(self) -> list[list[str]]:
        """Per-block streamed parameter names."""
        return [list(store.params) for store in self._block_stores]

    @property
    def streamed_buffer_names_by_block(self) -> list[list[str]]:
        """Per-block streamed buffer names."""
        return [list(store.buffers) for store in self._block_stores]

    @property
    def param_names(self) -> frozenset[str]:
        """Externally addressable streamed parameter names."""
        names = {
            _streamed_param_name(self.blocks_path, block_idx, local_name)
            for block_idx, store in enumerate(self._block_stores)
            for local_name in store.params
        }
        return frozenset(names)

    @property
    def buffer_names(self) -> frozenset[str]:
        """Externally addressable streamed buffer names."""
        names = {
            _streamed_param_name(self.blocks_path, block_idx, local_name)
            for block_idx, store in enumerate(self._block_stores)
            for local_name in store.buffers
        }
        return frozenset(names)

    @property
    def cache_bytes(self) -> int:
        return sum(store.cache_bytes for store in self._block_stores)

    @property
    def has_trainables(self) -> bool:
        return any(store.has_trainables for store in self._block_stores)

    def resolve_blocks(self, model: nn.Module) -> list[nn.Module]:
        """Resolve this store's blocks path on ``model``."""
        return _resolve_blocks(model, self.blocks_path)

    def bind(
        self,
        model: nn.Module,
    ) -> StreamedComponent:
        """Bind this store's per-block backing bytes to ``model``."""
        block_list = self.resolve_blocks(model)
        if len(block_list) != len(self._block_stores):
            raise ValueError(
                "StreamedComponentStore.bind() block count mismatch: "
                f"store has {len(self._block_stores)}, got {len(block_list)}."
            )
        instances = [
            store.bind(block)
            for store, block in zip(self._block_stores, block_list, strict=True)
        ]
        return StreamedComponent(
            instances,
            blocks_to_swap=self.blocks_to_swap,
            prefetch_count=self.prefetch_count,
            cyclic=self.cyclic,
            name=self.blocks_path,
        )


# ---------------------------------------------------------------------------
# StreamedComponent — public block-streaming primitive
# ---------------------------------------------------------------------------


class StreamedComponent:
    """Streams a single block list between pinned CPU and CUDA.

    The sharp, low-level streaming primitive. Manages bound block
    instances whose owned params and buffers are pinned to CPU and
    streams them to CUDA via forward-pre hooks on :meth:`activate`.
    CPU activation is pass-through over that pinned host-backed state:
    no pool, no hooks, no copies. Frozen params use parameter
    replacement; trainable params keep Parameter identity and swap only
    ``.data``.
    Does not touch parent modules, sibling modules, or out-of-block
    trainable parameters — those are the composer's responsibility.

    A :class:`StreamedComponent` is a *component* meant to be composed
    inside a :class:`~torch_offload.model_offloader.ModelOffloader`.
    It deliberately does NOT implement
    :class:`~torch_offload.protocols.ModelStrategy` (its
    :meth:`activate` returns ``None`` because it doesn't own the
    model). For top-level use, build a model binding via
    :class:`~torch_offload.model_offloader.ModelOffloader`.

    Lifecycle is uniform with :class:`PinnedComponent`: store construction
    pins (so ``cache_bytes`` is final at construction time, ready
    for :class:`~torch_offload.model_cache.ModelCache` admission), and
    ``activate`` brings to CUDA or marks CPU active, ``deactivate`` returns state to
    pinned CPU and removes hooks. There is no ``close()``; pinned
    memory in module state is freed when the caller drops the
    binding and model references.

    **Calling deactivate() before dropping the binding is preferred**
    — it removes the forward hooks (cleaner model state) and reverts
    GPU-resident blocks back to pinned CPU. The hook closure uses
    ``weakref.ref(self)`` so dropping the binding without deactivate
    is non-fatal: the orphaned hooks remain installed on the model
    but no-op; resident blocks stay on GPU until the model itself is
    dropped. Forward calls through still-resident blocks work; calls
    through previously-evicted blocks find pinned-CPU tensors (slow but
    functional).

    Instances are usually created by binding a
    :class:`StreamedComponentStore` to a compatible model.

    Parameters
    ----------
    block_instances:
        The concrete bound block instances.
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
        Optional model path for the streamed block list. When set,
        :attr:`param_names` and name-based post-copy hook registration
        use names like ``"blocks.3.weight"``. When omitted, standalone
        streamers use ``"3.weight"``.
        Trainable streaming requires activation checkpointing on every
        block (the ``.data`` swap bypasses autograd's version-counter
        check). The streamer doesn't enforce that precondition itself —
        :class:`ModelOffloader` does.
    """

    def __init__(
        self,
        block_instances: Sequence[PinnedModuleInstance],
        *,
        blocks_to_swap: int,
        prefetch_count: int = 2,
        cyclic: bool = False,
        name: str | None = None,
    ) -> None:
        self._block_instances = list(block_instances)
        self._blocks = [instance.module for instance in self._block_instances]
        self._active_device: torch.device | None = None
        self._blocks_to_swap = blocks_to_swap
        self._prefetch_count = prefetch_count
        self._cyclic = cyclic
        self._log_label = _streamed_log_label(name, len(self._blocks))
        self._param_name_to_block_param = _build_param_name_index(
            self._block_instances,
            name,
        )
        self._buffer_name_to_block_buffer = _build_buffer_name_index(
            self._block_instances,
            name,
        )
        self._param_names = frozenset(self._param_name_to_block_param)
        self._buffer_names = frozenset(self._buffer_name_to_block_buffer)
        self._move_trainable_grads_to(torch.device("cpu"))

        # Active resources allocated on CUDA activate().
        self._pool: _PinnedModuleTargetPool | None = None
        self._block_to_target: dict[int, int] = {}
        self._pool_config: tuple[int, torch.device] | None = None
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
    def blocks(self) -> tuple[nn.Module, ...]:
        """Bound block modules managed by this component."""
        return tuple(self._blocks)

    @property
    def streamed_param_names_by_block(self) -> list[list[str]]:
        """Per-block streamed parameter names."""
        return [
            list(instance.params)
            for instance in self._block_instances
        ]

    @property
    def streamed_buffer_names_by_block(self) -> list[list[str]]:
        """Per-block streamed buffer names."""
        return [
            list(instance.buffers)
            for instance in self._block_instances
        ]

    @property
    def param_names(self) -> frozenset[str]:
        """Externally addressable streamed parameter names."""
        return self._param_names

    @property
    def buffer_names(self) -> frozenset[str]:
        """Externally addressable streamed buffer names."""
        return self._buffer_names

    @property
    def has_trainables(self) -> bool:
        return any(instance.has_trainables for instance in self._block_instances)

    def register_post_copy_hook(
        self,
        name: str,
        hook: PostCopyHook,
    ) -> PostCopyHookHandle:
        """Register a hook after this component copies ``name`` to GPU.

        Package-internal: used by :class:`ModelOffloader` for merge-mode
        LoRA. Mirrors PyTorch's hook registration pattern by returning a
        handle whose :meth:`remove` method unregisters the hook.
        """
        instance, name = self._resolve_param_name(name)
        return instance.register_post_copy_hook(name, hook)

    def post_copy_hook_key(self, name: str) -> int:
        """Stable hook/dedup key for a streamed parameter."""
        instance, name = self._resolve_param_name(name)
        return instance.post_copy_hook_key(name)

    def _resolve_param_name(
        self,
        name: str,
    ) -> tuple[PinnedModuleInstance, str]:
        ref = self._param_name_to_block_param.get(name)
        if ref is None:
            raise ValueError(f"param name {name!r} is not owned by this streamer")
        block_idx, local_name = ref
        return self._resolve_block_param(block_idx, local_name)

    def _resolve_buffer_name(
        self,
        name: str,
    ) -> tuple[PinnedModuleInstance, str]:
        ref = self._buffer_name_to_block_buffer.get(name)
        if ref is None:
            raise ValueError(f"buffer name {name!r} is not owned by this streamer")
        block_idx, local_name = ref
        return self._resolve_block_buffer(block_idx, local_name)

    def _resolve_block_param(
        self,
        block_idx: int,
        name: str,
    ) -> tuple[PinnedModuleInstance, str]:
        if block_idx < 0 or block_idx >= len(self._block_instances):
            raise ValueError(
                f"streamed block index {block_idx} is out of range"
            )
        instance = self._block_instances[block_idx]
        if name not in instance.params:
            raise ValueError(
                f"param name {name!r} is not owned by streamed block {block_idx}"
            )
        return instance, name

    def _resolve_block_buffer(
        self,
        block_idx: int,
        name: str,
    ) -> tuple[PinnedModuleInstance, str]:
        if block_idx < 0 or block_idx >= len(self._block_instances):
            raise ValueError(
                f"streamed block index {block_idx} is out of range"
            )
        instance = self._block_instances[block_idx]
        if name not in instance.buffers:
            raise ValueError(
                f"buffer name {name!r} is not owned by streamed block {block_idx}"
            )
        return instance, name

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        if device is not None:
            return canonical_device(device)
        raise ValueError(
            "StreamedComponent.activate() requires a device"
        )

    def _require_active_device(self) -> torch.device:
        device = self._active_device
        if device is None:
            raise RuntimeError("StreamedComponent has no active device")
        return device

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self, device: torch.device | str | None = None) -> None:
        """Activate the block list on ``device``.

        CUDA activation uses the streaming path: GPU target pool, CUDA
        stream/events, prefetch executor, and forward hooks. CPU
        activation is pass-through over the pinned host-backed state.
        The composite's :meth:`activate` returns the model — this
        method returns ``None`` because the streamer doesn't own one.

        **Lifecycle is caller's responsibility.** Calling activate()
        twice without an intervening deactivate() raises before hooks or
        block pools are installed.

        **Activation failure semantics:** if activation fails midway,
        the streamer is left in an undefined partial state. Retrying
        activation on that streamer is unsupported; the caller's only
        supported cleanup path is :meth:`deactivate`, which idempotently
        tears down whatever was allocated."""
        # Hard-guard against the documented "don't activate twice"
        # case. Without this, a double-activate would double-install
        # forward-pre hooks (silent grad doubling) and stack a second
        # target pool on top of an active one.
        if self._active_device is not None:
            raise RuntimeError(
                "StreamedComponent.activate() called while already "
                "active. Deactivate first, or check for a leaked "
                "context manager."
            )
        active_device = self._resolve_device(device)
        if active_device.type == "cpu":
            self._activate_cpu_resolved()
            return
        if active_device.type != "cuda":
            raise ValueError(
                "StreamedComponent.activate() supports CUDA or CPU; "
                f"got {active_device}."
            )
        self._activate_cuda_resolved(active_device)

    def _activate_cpu_resolved(self) -> None:
        self._active_device = torch.device("cpu")

    def _activate_cuda_resolved(self, active_device: torch.device) -> None:
        num_blocks = len(self._blocks)
        num_resident = num_blocks - self._blocks_to_swap
        num_gpu_targets = num_resident + self._prefetch_count

        self._active_device = active_device
        self._tracker = _BlockTracker()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._stream = torch.cuda.Stream(device=active_device, priority=-1)
        self._pending = {}
        self._prefetch_events = {i: torch.cuda.Event() for i in range(num_blocks)}
        self._last_idx = -1

        self._activate_pool(num_gpu_targets, active_device)
        self._move_trainable_grads_to(active_device)

        for block_idx in range(min(num_resident, num_blocks)):
            self._load_block(block_idx)
            self._tracker.mark_on_gpu(block_idx)

        self._register_hooks(num_resident)
        self.reset_peak()

        logger.info(
            f"{self._log_label} active: {self._blocks_to_swap}/{num_blocks} on CPU, "
            f"{num_resident} resident on GPU, prefetch={self._prefetch_count}, "
            f"gpu_pool_targets={num_gpu_targets}"
        )

    def deactivate(self) -> None:
        """Tear down active resources idempotently — safe to call
        before activate or multiple times. Every step in
        ``_teardown_active_resources`` null-checks; partial state
        from a failed activate cleans up correctly. Drop the
        binding reference after deactivate to release pinned
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

    # ------------------------------------------------------------------
    # Optimizer-step boundary
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def optimizer_step(self) -> Iterator[None]:
        """Context manager for the streamed-trainable optimizer boundary.

        On CUDA activation, this brings trainable ``.data`` to GPU
        around the optimizer-step boundary. On CPU activation, it is a
        guarded no-op because trainable ``.data`` is already resident in
        the pinned host-backed module state.

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
             every allocated target). Raises if a prefetch errored —
             eviction still ran first so the streamer is in a
             consistent baseline.
          2. On the streamer's private ``self._stream``, H2D each
             trainable's pinned ``.data`` to fresh instance-owned GPU
             target. Each H2D registers a rollback (repoint ``.data`` at
             pinned) on an :class:`~contextlib.ExitStack`.
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

        On CUDA, the target pool's pre-warmed state is lost (next forward
        re-loads from pinned), but that cost is dominated by the
        forward + backward of the next iteration.
        """
        if self._executor is None:
            if self._active_device is not None and self._active_device.type == "cpu":
                if self._optimizer_step_active:
                    raise RuntimeError(
                        "StreamedComponent.optimizer_step() does not support "
                        "reentrant entry."
                    )
                self._optimizer_step_active = True
                try:
                    yield
                finally:
                    self._optimizer_step_active = False
                return
            raise RuntimeError(
                "StreamedComponent.optimizer_step() called on inactive "
                "streamer. Use it inside the offloader's context "
                "manager, between backward and the next forward."
            )
        if self._optimizer_step_active:
            raise RuntimeError(
                "StreamedComponent.optimizer_step() does not support "
                "reentrant entry. A nested optimizer-step boundary would "
                "scatter the outer step's stale pinned bytes on top of "
                "the inner update."
            )
        if not self.has_trainables:
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
                loaded = self._load_trainables_for_step(
                    target, step_stream, stack,
                )

                try:
                    yield
                finally:
                    self._scatter_trainables_after_step(
                        loaded, step_stream, target,
                    )
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
    # Trainable helpers
    # ------------------------------------------------------------------

    def _load_trainables_for_step(
        self,
        device: torch.device,
        step_stream: torch.cuda.Stream,
        stack: contextlib.ExitStack,
    ) -> list[_LoadedTrainableBlock]:
        """Move trainable ``.data`` to ``device`` for optimizer.step()."""
        # Each block load registers a rollback that restores the block
        # instance to pinned host storage. ExitStack unwinds
        # these on the way out whether enter failed mid-loop, the body
        # exited cleanly, or the body raised after yield.
        loaded: list[_LoadedTrainableBlock] = []
        with torch.cuda.stream(step_stream):
            for instance in self._block_instances:
                if not instance.has_trainables:
                    continue
                stack.callback(instance.restore_pinned)
                trainable_target = instance.allocate_target(
                    device,
                    param_names=instance.trainable_param_names,
                    buffer_names=(),
                )
                instance.load_to_target(trainable_target, non_blocking=True)
                _move_instance_trainable_grads_to(instance, device)
                loaded.append((instance, trainable_target))
        # User's current stream now waits for step_stream's H2D. After
        # this point the optimizer can safely read param.data on its own
        # stream.
        torch.cuda.current_stream(device).wait_stream(step_stream)
        return loaded

    def _scatter_trainables_after_step(
        self,
        loaded: list[_LoadedTrainableBlock],
        step_stream: torch.cuda.Stream,
        device: torch.device,
    ) -> None:
        """Copy optimizer-updated trainable ``.data`` back to pinned CPU."""
        # Scatter on body-exit (clean OR exception). On body exception
        # the optimizer may have mutated some params before raising;
        # preserve that partial state rather than silently rolling back
        # to pre-step pinned bytes. The blocking copy + sync guarantees
        # the next prefetch reads stable bytes.
        step_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(step_stream):
            for instance, trainable_target in loaded:
                instance.copy_trainables_from_target(
                    trainable_target,
                    non_blocking=False,
                )
        step_stream.synchronize()
        # ExitStack unwinds after this returns: each materialized
        # block instance restores trainable .data to pinned CPU storage,
        # releasing the optimizer-step GPU allocation.

    def _move_trainable_grads_to(self, device: torch.device) -> None:
        """Move each trainable's ``.grad`` (if any) to ``device``.

        During backward, PyTorch's native ``AccumulateGrad`` writes
        grads on the param's data device, which is GPU at that point
        because the ``.data`` swap in :meth:`PinnedModuleInstance.load_to_target`
        repointed ``.data`` at target storage. Eviction restores ``.data``
        to pinned CPU, but ``.grad`` keeps living wherever AccumulateGrad
        placed it.
        """
        for instance in self._block_instances:
            _move_instance_trainable_grads_to(instance, device)

    # ------------------------------------------------------------------
    # Pool and block movement
    # ------------------------------------------------------------------

    def _activate_pool(self, num_gpu_targets: int, device: torch.device) -> None:
        if self._pool_config is not None:
            existing = self._pool_config
            if existing != (num_gpu_targets, device):
                raise ValueError(
                    f"StreamedComponent pool already activated with "
                    f"{existing}; cannot re-activate with ({num_gpu_targets}, "
                    f"{device}). Call _deactivate_pool() first."
                )
            return
        if num_gpu_targets <= 0:
            raise ValueError(
                f"num_gpu_targets must be > 0, got {num_gpu_targets}. "
                "num_resident is always >= 1 by construction; this only "
                "fires when prefetch_count is negative."
            )

        # Pool template comes from block 0. The constructor's layout
        # check has already verified every other block matches.
        self._pool = _PinnedModuleTargetPool(
            self._block_instances[0],
            num_gpu_targets,
            device,
        )
        self._pool_config = (num_gpu_targets, device)

    def _deactivate_pool(self) -> None:
        self._pool = None
        self._block_to_target.clear()
        self._pool_config = None

    def _load_block(
        self,
        block_idx: int,
        *,
        non_blocking: bool = False,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        assert self._pool is not None, "_activate_pool not called"

        instance = self._block_instances[block_idx]

        target_idx = self._block_to_target.get(block_idx)
        if target_idx is None:
            target_idx = self._pool.acquire()
            self._block_to_target[block_idx] = target_idx

        self._pool.wait_if_needed(target_idx, stream)
        target = self._pool.target(target_idx)
        instance.load_to_target(
            target,
            run_post_copy_hooks=True,
            non_blocking=non_blocking,
        )

    def _release_block(self, block_idx: int) -> None:
        self._block_instances[block_idx].restore_pinned()
        if self._pool is None:
            return
        target_idx = self._block_to_target.pop(block_idx, None)
        if target_idx is not None:
            self._pool.release(target_idx)

    def _evict_allocated_blocks(self) -> None:
        """Evict every block that currently holds a pool target.

        ``_block_to_target`` is the source of truth for target allocation:
        a block is recorded there as soon as :meth:`_load_block`
        (foreground or prefetch) acquires its target, and removed only
        by :meth:`_release_block`. Iterating it catches both
        currently-resident blocks and pending-prefetch blocks whose H2D
        may still be in flight, so teardown and optimizer_step don't
        need to reconcile tracker and pending-future state.

        Caller is responsible for stream/event synchronization before
        the eviction so in-flight DMA into the target bytes has settled.
        """
        for block_idx in list(self._block_to_target.keys()):
            self._release_block(block_idx)

    def _mark_compute_done(
        self, block_idx: int, event: torch.cuda.Event,
    ) -> None:
        assert self._pool is not None, "_activate_pool not called"
        target_idx = self._block_to_target.get(block_idx)
        if target_idx is not None:
            self._pool.set_compute_event(target_idx, event)

    # ------------------------------------------------------------------
    # Drain and teardown
    # ------------------------------------------------------------------

    def _drain_and_evict_all(self) -> BaseException | None:
        """Quiesce streaming back to a baseline state.

        Drains pending prefetch futures (waits for completion),
        synchronizes the prefetch stream so any in-flight H2D has
        settled device-side, then evicts every block currently holding
        a pool target via the streamer's ``_block_to_target`` source of
        truth, and clears tracker + pending bookkeeping.

        Idempotent — safe to call when no resources are active and
        safe to call multiple times. Returns the first prefetch
        exception encountered (or ``None``) so the caller can choose
        to surface it. We don't raise here so eviction still runs
        after a failed prefetch; without this, a transient prefetch
        error would leak the GPU target it had partially populated.

        Used by both :meth:`_teardown_active_resources` (full
        deactivate) and :meth:`optimizer_step` (mid-cycle quiesce
        before optimizer step). Centralizing it removes the bug
        class where the two paths' bookkeeping diverged — optimizer_step
        used to walk ``_tracker._on_gpu`` and miss pending prefetch
        targets, leaking them across step boundaries.
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

        self._evict_allocated_blocks()

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
        self._move_trainable_grads_to(torch.device("cpu"))
        if active_device is None or active_device.type == "cuda":
            self._deactivate_pool()

        self._tracker = None
        self._last_idx = -1
        self._active_device = None

        return first_prefetch_exc

    # ------------------------------------------------------------------
    # Prefetch and forward hooks
    # ------------------------------------------------------------------

    def _evict_one(self, protected: set[int], compute_event: object | None = None) -> None:
        assert self._tracker is not None
        victim = self._tracker.pick_victim(protected=protected)
        if compute_event is not None:
            self._mark_compute_done(victim, cast(torch.cuda.Event, compute_event))
        self._release_block(victim)
        self._tracker.mark_on_cpu(victim)

    def _do_prefetch(self, idx: int) -> None:
        assert self._stream is not None
        with torch.cuda.stream(self._stream):
            self._load_block(idx, non_blocking=True, stream=self._stream)
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
        future = self._pending.pop(idx, None)
        if future is not None:
            future.result()
            ev = self._prefetch_events[idx]
            if not ev.query():
                torch.cuda.current_stream(self._require_active_device()).wait_event(ev)
            self._tracker.mark_on_gpu(idx)
            return

        if not self._tracker.is_on_gpu(idx):
            self._load_block(idx)
            self._tracker.mark_on_gpu(idx)

    def _before_block_forward(
        self,
        idx: int,
        *,
        num_resident: int,
        max_on_gpu: int,
        prefetch_count: int,
        cyclic: bool,
        num_blocks: int,
        wrap_threshold: int,
    ) -> None:
        tracker = self._tracker
        if tracker is None:
            return

        pending = self._pending
        if tracker.is_on_gpu(idx):
            tracker.touch(idx)
        else:
            compute_event = torch.cuda.current_stream(
                self._require_active_device()
            ).record_event()
            while len(tracker._on_gpu) >= num_resident:
                protected = {idx} | set(pending.keys())
                self._evict_one(protected, compute_event)
            self._ensure_on_gpu(idx)

        # Direction inference. In cyclic mode, a large index jump
        # (|Delta| > num_blocks/2) is iteration wraparound, not a
        # reversal: keep the same forward/backward sense and let
        # prefetch indices wrap modulo num_blocks so the next
        # iteration's leading blocks get streamed proactively.
        last = self._last_idx
        self._last_idx = idx
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
                target %= num_blocks
            self._submit_prefetch(target, max_on_gpu)

        total = len(tracker._on_gpu) + len(pending)
        tracker.peak_gpu_blocks = max(tracker.peak_gpu_blocks, total)

    def _register_hooks(self, num_resident: int) -> None:
        idx_map: dict[int, int] = {id(block): idx for idx, block in enumerate(self._blocks)}
        prefetch_count = self._prefetch_count  # capture as local — no `self` ref in closure
        max_on_gpu = num_resident + prefetch_count
        cyclic = self._cyclic
        num_blocks = len(self._blocks)
        wrap_threshold = num_blocks // 2  # |Δidx| > this counts as wraparound
        # weakref breaks the cycle: block → hook → closure → streamer
        # → _blocks → block. Without it, dropping the binding without
        # first calling deactivate() would keep everything alive until
        # Python's cycle collector runs (not refcount-based GC). The
        # weak ref lets refcount immediately free the binding when the
        # caller drops it; orphaned hooks no-op safely.
        self_ref = weakref.ref(self)

        def _pre_hook(_module: nn.Module, _args: tuple[object, ...], *, idx: int) -> None:
            streamer = self_ref()
            if streamer is None:
                # Strategy was dropped without deactivate. Hook is
                # orphaned — no-op. Forward through this block uses
                # whatever the registry currently holds: GPU param if it
                # was resident at drop-time, instance cpu_param if it
                # had been evicted. Both are functional (CPU forward
                # is slower but works on pinned tensors).
                return
            streamer._before_block_forward(
                idx,
                num_resident=num_resident,
                max_on_gpu=max_on_gpu,
                prefetch_count=prefetch_count,
                cyclic=cyclic,
                num_blocks=num_blocks,
                wrap_threshold=wrap_threshold,
            )

        for block in self._blocks:
            idx = idx_map[id(block)]
            h = block.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            self._hooks.append(h)


__all__ = [
    "StreamedComponent",
    "StreamedComponentStore",
]
