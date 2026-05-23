"""Unified CUDA offload strategy with optional LoRA application.

Supports whole-model pinned bulk offload or block streaming, with optional
per-weight LoRA application in both modes.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import nn

from ._devices import canonical_device
from .lora import LoRA, LoRARouteHandle, LoRATransform
from .module_names import (
    buffer_names,
    canonical_param_name,
    parameter_names,
    resolve_parent_leaf,
)
from .pinned_component import PinnedComponent, PinnedComponentStore
from .protocols import ModelStrategyComponent
from .streamed_component import StreamedComponent, StreamedComponentStore
from .tensor_adapter_registry import select_adapter

logger = logging.getLogger(__name__)

LoraMode = Literal["merge", "routed"]
_LoraFactorRef = tuple[torch.Tensor, torch.Tensor, float]
_LoraParamMap = dict[str, list[_LoraFactorRef]]


class _RemovableHook(Protocol):
    def remove(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class _ModelOffloaderBinding:
    pinned_component: PinnedComponent | None
    streamed_components: list[StreamedComponent]
    block_groups: list[list[nn.Module]]
    components: list[ModelStrategyComponent]
    lora_param_names: frozenset[str]


@dataclass(frozen=True, slots=True)
class _ModelOffloaderStore:
    pinned_component_store: PinnedComponentStore | None
    streamed_component_stores: tuple[StreamedComponentStore, ...]
    stream_trainable_weights: bool

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        layer_paths: Sequence[str],
        blocks_to_swap: int | Sequence[int] | None,
        prefetch_count: int | Sequence[int],
        cyclic: bool,
        stream_trainable_weights: bool,
        include_param_names: Iterable[str] | None,
        include_buffer_names: Iterable[str] | None,
    ) -> _ModelOffloaderStore:
        (
            streamed_component_stores,
            streamed_param_names,
            streamed_buffer_names,
        ) = _build_streamed_component_stores(
            model,
            layer_paths=layer_paths,
            blocks_to_swap=blocks_to_swap,
            prefetch_count=prefetch_count,
            cyclic=cyclic,
            stream_trainable_weights=stream_trainable_weights,
        )
        pinned_component_store = _build_pinned_component_store(
            model,
            streamed_param_names=streamed_param_names,
            streamed_buffer_names=streamed_buffer_names,
            include_param_names=include_param_names,
            include_buffer_names=include_buffer_names,
        )
        return cls(
            pinned_component_store=pinned_component_store,
            streamed_component_stores=streamed_component_stores,
            stream_trainable_weights=stream_trainable_weights,
        )

    @property
    def cache_bytes(self) -> int:
        pinned_bytes = (
            0
            if self.pinned_component_store is None
            else self.pinned_component_store.cache_bytes
        )
        streamed_bytes = sum(
            store.cache_bytes for store in self.streamed_component_stores
        )
        return pinned_bytes + streamed_bytes

    def bind(self, model: nn.Module) -> _ModelOffloaderBinding:
        streamed_components: list[StreamedComponent] = []
        block_groups: list[list[nn.Module]] = []
        for store in self.streamed_component_stores:
            block_groups.append(store.resolve_blocks(model))
            streamed_components.append(store.bind(model))

        pinned_component = (
            None
            if self.pinned_component_store is None
            else self.pinned_component_store.bind(model)
        )
        components = _compose_components(pinned_component, streamed_components)
        lora_param_names: set[str] = set()
        if pinned_component is not None:
            lora_param_names.update(pinned_component.param_names)
        for streamed_component in streamed_components:
            lora_param_names.update(streamed_component.param_names)
        return _ModelOffloaderBinding(
            pinned_component=pinned_component,
            streamed_components=streamed_components,
            block_groups=block_groups,
            components=components,
            lora_param_names=frozenset(lora_param_names),
        )


__all__ = [
    "ModelOffloader",
]


def _routed_factor_dtype(module: nn.Module) -> torch.dtype:
    """Compute dtype to cast routed-mode LoRA factors into.

    For routed LoRA, the hook adds ``x @ A.T @ B.T`` to the layer's
    output, so factors must land in the same dtype the layer produces.

    Probe order:

    - ``module.compute_dtype`` — BitsAndBytes ``Linear4bit`` and
      similar modules that carry an explicit compute-dtype attribute.
    - ``select_adapter(module.weight.data).compute_dtype(...)`` — plain
      tensors return ``weight.dtype``; structured/quantized wrappers can
      return their logical matmul dtype instead of packed storage dtype.
    """
    compute = getattr(module, "compute_dtype", None)
    if compute is not None:
        return compute
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError(
            f"Routed LoRA mode requires a tensor-like weight on "
            f"{type(module).__name__}; got {type(weight).__name__}"
        )
    return select_adapter(weight.data).compute_dtype(weight.data)


class ModelOffloader:
    """Move a whole model or streamed block groups between pinned CPU and
    CUDA, with optional LoRA merge and trainable-parameter support.

    When ``layers_attr`` is omitted, CUDA activation bulk-copies every
    managed parameter and buffer to CUDA. When ``layers_attr`` is set,
    CUDA activation streams the selected block groups plus component-level
    movement for non-streamed state. CPU activation is pass-through over
    the pinned host-backed module state.

    Composes :class:`PinnedComponent` (non-streamed params and buffers)
    and one or more :class:`StreamedComponent`\\ s internally. LoRA requests
    are recorded via
    :meth:`set_loras` and applied on :meth:`activate`, where merge mode
    installs activation-scoped post-copy hooks so the merge fires
    immediately after each CPU->GPU weight copy — no separate merge
    strategy needed.

    Training
    --------
    Training through streamed blocks **requires activation
    checkpointing on each block** — wrap call sites in
    :func:`torch.utils.checkpoint.checkpoint`, or call
    ``model.gradient_checkpointing_enable()`` on a HuggingFace model.
    Without it, ``loss.backward()`` raises ``RuntimeError: ... has
    been modified by an inplace operation`` on the first target reuse.

    Why: autograd saves a reference to each ``Linear``'s weight
    tensor at forward time and records its version counter. Streaming
    is a sequence of in-place ``copy_`` writes into a fixed pool of
    GPU target tensors — every load bumps the target tensor's version,
    invalidating any previously-saved reference into that target.
    Checkpointing makes each block's internal forward run under
    ``no_grad`` (no internal tensors saved); when backward arrives,
    PyTorch re-runs the block's forward with grad enabled, building
    a fresh autograd graph whose saved references only live within
    that one block's recompute-then-backward window. Target reuse
    outside that window is then safe.

    For **frozen-only** streamed blocks (training touches only
    out-of-block trainables), CUDA :meth:`activate` emits a one-time
    warning if no HuggingFace ``gradient_checkpointing`` flag is
    detected — the failure mode without checkpointing is a loud
    ``RuntimeError`` from autograd's saved-tensor check, so a warning
    suffices.

    By default, trainable params are not streamed through the block
    residency pool. They are managed by :class:`PinnedComponent`, stay
    GPU-resident while the offload strategy is active on CUDA, and must be
    updated inside :meth:`optimizer_step` so CUDA updates are copied
    back to the pinned CPU cache. CPU activation leaves them in the
    host-backed module state.

    Pass ``stream_trainable_weights=True`` to stream in-block
    trainable parameter data through the CUDA block target pool. In that
    mode, CUDA :meth:`activate` *raises* during training if no
    ``gradient_checkpointing`` flag is detected. The failure mode
    here is **silent gradient corruption** rather than a loud
    error — the ``.data`` swap path used for trainable streaming
    bypasses autograd's version-counter check, so we hard-guard
    the precondition. Pass ``skip_checkpointing_check=True`` to suppress
    the check if you wrap blocks manually via
    ``torch.utils.checkpoint.checkpoint`` at call sites (the
    wrapping is invisible from the module tree, so detection has
    false negatives). Callers passing this flag take responsibility
    for ensuring every streamed block that participates in training
    is wrapped.

    With ``stream_trainable_weights=True`` on CUDA,
    :meth:`optimizer_step` is the optimizer boundary: it materializes
    streamed trainable ``.data`` on GPU while an arbitrary PyTorch
    optimizer updates it, then copies the updated data back to pinned
    CPU. CPU activation makes :meth:`optimizer_step` a guarded no-op.
    Gradients are not streamed; PyTorch owns ``param.grad`` normally.

    Parameters
    ----------
    model:
        The model containing the block list(s). Managed tensors may start
        on CPU or CUDA; construction clones them directly into pinned
        CPU storage before activation.
    layers_attr:
        Optional dotted attribute path(s) to ``nn.ModuleList`` block
        list(s). When omitted, :class:`ModelOffloader` performs whole-model
        bulk pinning with no streamed block components. For PEFT-wrapped
        models, include the PEFT prefix (e.g.
        ``"base_model.model.transformer_blocks"``).
    blocks_to_swap:
        Per-group count of blocks to keep on CPU. Required when
        ``layers_attr`` is set. Single int (broadcast to all groups) or
        one int per group.
    prefetch_count:
        Per-group prefetch depth. Same broadcasting as *blocks_to_swap*.
    cyclic:
        Default ``False``. Forwarded to every :class:`StreamedComponent`.
        Set ``True`` for inference loops that iterate the model
        repeatedly (diffusion denoising, multi-step decoders); the
        prefetcher then treats end-of-iteration as wraparound and
        keeps streaming the next iteration's leading blocks. Leave
        ``False`` for single-shot inference or training.
    stream_trainable_weights:
        Default ``False`` skips trainable params in block streaming and
        manages them with :class:`PinnedComponent`. ``True`` streams
        in-block trainable parameter data with the block residency
        manager. In both modes, wrap optimizer updates in
        :meth:`optimizer_step` so trainable CUDA updates are copied back
        to pinned CPU storage.
    skip_checkpointing_check:
        Default ``False``. Suppresses the activate-time checkpointing
        guard/warning entirely. Pass ``True`` only if you wrap each
        streamed block that participates in training manually via
        ``torch.utils.checkpoint.checkpoint`` at call sites — that
        style is invisible from the module tree, so the auto-detect
        sees no flag and would raise. By passing ``True`` you take
        responsibility for ensuring every streamed block that
        participates in training is checkpointed; otherwise trainable
        streaming may silently corrupt gradients.
    is_block_checkpointed:
        Optional per-block predicate, called once per streamed block
        at :meth:`activate` to decide whether the block runs under
        activation checkpointing. Default detection reads the
        HuggingFace ``gradient_checkpointing`` boolean attribute on
        the block module. Pass a custom callable to support
        non-HF frameworks (DeepSpeed, custom training loops, etc.) —
        the predicate sees the block module and should return
        ``True`` iff that block runs under activation checkpointing.
        Ignored when ``skip_checkpointing_check=True``.
    include_param_names:
        Optional PyTorch parameter names to manage in whole-model mode.
        ``None`` manages all parameters. Only valid when ``layers_attr``
        is omitted.
    include_buffer_names:
        Optional PyTorch buffer names to manage in whole-model mode.
        ``None`` manages all registered buffers. Only valid when
        ``layers_attr`` is omitted.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        layers_attr: str | Sequence[str] | None = None,
        blocks_to_swap: int | Sequence[int] | None = None,
        prefetch_count: int | Sequence[int] = 2,
        cyclic: bool = False,
        stream_trainable_weights: bool = False,
        skip_checkpointing_check: bool = False,
        is_block_checkpointed: Callable[[nn.Module], bool] | None = None,
        include_param_names: Iterable[str] | None = None,
        include_buffer_names: Iterable[str] | None = None,
    ) -> None:
        layer_paths = _normalize_layer_paths(layers_attr)
        _validate_constructor_mode(
            layer_paths=layer_paths,
            blocks_to_swap=blocks_to_swap,
            include_param_names=include_param_names,
            include_buffer_names=include_buffer_names,
        )
        store = _ModelOffloaderStore.from_module(
            model,
            layer_paths=layer_paths,
            blocks_to_swap=blocks_to_swap,
            prefetch_count=prefetch_count,
            cyclic=cyclic,
            stream_trainable_weights=stream_trainable_weights,
            include_param_names=include_param_names,
            include_buffer_names=include_buffer_names,
        )
        binding = store.bind(model)

        self._model = model
        self._store = store
        self._active_device: torch.device | None = None
        self._components = binding.components
        self._pinned_component = binding.pinned_component
        self._streamed_components = binding.streamed_components
        if binding.pinned_component is not None:
            self._instance = binding.pinned_component._instance
        self._teardown_stack: contextlib.ExitStack | None = None
        self._lora_param_names = binding.lora_param_names
        self._lora_hook_handles: list[_RemovableHook] = []
        self._block_groups: list[list[nn.Module]] = binding.block_groups
        self._warned_about_checkpointing: bool = False
        self._stream_trainable_weights: bool = store.stream_trainable_weights
        self._skip_checkpointing_check: bool = skip_checkpointing_check
        self._is_block_checkpointed: Callable[[nn.Module], bool] = (
            is_block_checkpointed
            if is_block_checkpointed is not None
            else _hf_block_has_checkpointing_flag
        )
        # Configured LoRA request. set_loras() only records caller intent;
        # activate(device) groups targets and validates the requested
        # application path once the runtime context is known.
        self._loras: list[tuple[LoRA, float]] = []
        self._lora_mode: LoraMode = "merge"

    # ------------------------------------------------------------------ API

    def set_loras(
        self,
        loras: Sequence[tuple[LoRA, float]],
        *,
        mode: LoraMode = "merge",
    ) -> None:
        """Record LoRAs for the next activation cycle.

        Must be called while deactivated. Attachments are cleared
        automatically on :meth:`deactivate`, so callers must call
        ``set_loras`` before each :meth:`activate` if they want
        LoRA-augmented inference.

        ``mode``:

        - ``"merge"`` (default): on CUDA activation, installs a
          post-copy hook per matched target. The hook applies
          :class:`LoRATransform` immediately after the base weight is
          copied to GPU, so it rides along with the streaming cycle.
          Requires CUDA activation and a base-weight adapter that
          supports dense in-place ``addmm_`` or dequantize/requantize
          plus ``copy_into`` merge.
        - ``"routed"``: on activation, registers a forward hook on each
          matched parent module. Forward becomes
          ``y = base(x) + alpha * B * A * x``. Doesn't touch the base
          weight in place. Restricted to ``nn.Linear`` parents (the math
          assumes ``y = x @ W.T``); non-Linear parents raise. Shared
          weight storage is allowed because routed mode hooks the exact
          matched parent module instead of mutating weight bytes.
          Quantized bases work when the matched module still exposes the
          logical ``nn.Linear`` weight shape and either its adapter
          reports a logical compute dtype or the module exposes
          ``compute_dtype``. Packed formats whose parameter shape
          differs from the logical matmul weight need a per-format route
          layer.
        Routed mode requires activations to reach the hooked layer in
        the layer's compute dtype (or under autocast). Mixed-dtype
        inputs without autocast will error in the hook's matmul.

        Pass an empty sequence to clear all LoRAs (base-only forward).
        """
        if self._teardown_stack is not None:
            raise RuntimeError(
                "ModelOffloader.set_loras() requires the strategy "
                "to be inactive. Call deactivate() first."
            )
        if mode not in ("merge", "routed"):
            raise ValueError(
                f"set_loras mode must be 'merge' or 'routed', got {mode!r}"
            )
        configured_loras = list(loras)
        self._loras = configured_loras
        self._lora_mode = mode if configured_loras else "merge"

    def _group_lora_factors_by_param_name(
        self, loras: Sequence[tuple[LoRA, float]],
    ) -> _LoraParamMap:
        per_param: _LoraParamMap = {}
        total_targets = 0
        matched_targets = 0
        for lora, strength in loras:
            for target_key, (a, b) in lora.targets.items():
                total_targets += 1
                param_name = self._resolve_lora_param_name(target_key)
                if param_name is None:
                    continue
                matched_targets += 1
                per_param.setdefault(param_name, []).append(
                    (a, b, strength)
                )

        if matched_targets < total_targets:
            sample_lora = sorted(next(iter(loras))[0].targets)[:3]
            sample_index = sorted(self._lora_param_names)[:3]
            logger.warning(
                "set_loras matched %d/%d targets. "
                "Sample LoRA keys: %s ... Sample index keys: %s ...",
                matched_targets, total_targets, sample_lora, sample_index,
            )
        else:
            logger.debug("set_loras matched %d/%d targets", matched_targets, total_targets)

        return per_param

    def _resolve_lora_param_name(self, target_key: str) -> str | None:
        if target_key in self._lora_param_names:
            return target_key

        canonical_key = canonical_param_name(target_key)
        matches = [
            name
            for name in self._lora_param_names
            if canonical_param_name(name) == canonical_key
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _register_lora_hooks(
        self, active_device: torch.device, targets: _LoraParamMap,
    ) -> None:
        self._clear_active_lora_hooks()
        try:
            if self._lora_mode == "merge":
                self._register_merge_lora_hooks(active_device, targets)
            else:
                self._register_routed_lora_hooks(active_device, targets)
        except BaseException:
            self._clear_active_lora_hooks()
            raise

    def _register_merge_lora_hooks(
        self, active_device: torch.device, targets: _LoraParamMap,
    ) -> None:
        if active_device.type != "cuda":
            raise ValueError(
                "ModelOffloader merge mode requires CUDA activation; "
                f"got {active_device}. Use set_loras(..., mode='routed') "
                "for CPU activation."
            )

        for param_name, refs in targets.items():
            transform = LoRATransform(refs)
            handle = self._register_post_copy_hook(
                param_name, transform.apply,
            )
            self._lora_hook_handles.append(handle)

    def _register_routed_lora_hooks(
        self,
        active_device: torch.device,
        targets: _LoraParamMap,
    ) -> None:
        for param_name, refs in targets.items():
            parent, _leaf = resolve_parent_leaf(self._model, param_name)
            if not isinstance(parent, nn.Linear):
                raise ValueError(
                    f"Routed LoRA mode requires nn.Linear targets; "
                    f"target {param_name!r} has parent module of "
                    f"type {type(parent).__name__}. Use mode='merge' "
                    f"for non-Linear targets, or wrap the model with "
                    f"PEFT for richer per-type routing."
                )
            handle = LoRARouteHandle(
                parent, refs, active_device,
                dtype=_routed_factor_dtype(parent),
            )
            self._lora_hook_handles.append(handle)

    def _register_post_copy_hook(
        self,
        param_name: str,
        hook: Callable[[nn.Parameter], None],
    ) -> _RemovableHook:
        component = self._component_for_param_name(param_name)
        return component.register_post_copy_hook(param_name, hook)

    def register_post_copy_hook(
        self,
        param_name: str,
        hook: Callable[[nn.Parameter], None],
    ) -> _RemovableHook:
        """Register a hook after the owning component copies ``param_name``."""
        return self._register_post_copy_hook(param_name, hook)

    def post_copy_hook_key(self, param_name: str) -> int:
        """Stable hook/dedup key for a managed parameter name."""
        component = self._component_for_param_name(param_name)
        return component.post_copy_hook_key(param_name)

    def _component_for_param_name(
        self,
        param_name: str,
    ) -> PinnedComponent | StreamedComponent:
        if (
            self._pinned_component is not None
            and param_name in self._pinned_component.param_names
        ):
            return self._pinned_component
        for streamed_component in self._streamed_components:
            if param_name in streamed_component.param_names:
                return streamed_component
        raise KeyError(f"param name {param_name!r} is not managed by this ModelOffloader")

    def _clear_active_lora_hooks(self) -> None:
        while self._lora_hook_handles:
            self._lora_hook_handles.pop().remove()

    def _clear_loras(self) -> None:
        self._clear_active_lora_hooks()
        self._loras = []
        self._lora_mode = "merge"

    # ------------------------------------------------- ModelStrategy interface

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def value(self) -> nn.Module:
        return self._model

    @property
    def param_names(self) -> frozenset[str]:
        """Pinned parameter names managed by the non-streamed component."""
        if self._pinned_component is None:
            return frozenset()
        return self._pinned_component.param_names

    @property
    def buffer_names(self) -> frozenset[str]:
        """Pinned buffer names managed by the non-streamed component."""
        if self._pinned_component is None:
            return frozenset()
        return self._pinned_component.buffer_names

    @property
    def cache_bytes(self) -> int:
        return sum(c.cache_bytes for c in self._components)

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        if device is not None:
            return canonical_device(device)
        raise ValueError(
            "ModelOffloader.activate() requires a device; pass "
            "activate(device) or use this strategy through "
            "ModelCache.use(..., device=...)"
        )

    def activate(self, device: torch.device | str | None = None) -> None:
        if self._active_device is not None:
            raise RuntimeError(
                "ModelOffloader.activate() called while already active "
                f"on {self._active_device}. Deactivate first, or check "
                "for a leaked context manager."
            )
        active_device = self._resolve_device(device)
        if active_device.type not in ("cpu", "cuda"):
            raise ValueError(
                "ModelOffloader.activate() supports CUDA or CPU; "
                f"got {active_device}."
        )
        if active_device.type == "cuda":
            self._enforce_checkpointing_for_trainable_streaming()
            self._warn_if_training_without_checkpointing()
        self._active_device = active_device
        try:
            with contextlib.ExitStack() as stack:
                targets = (
                    self._group_lora_factors_by_param_name(self._loras)
                    if self._loras
                    else None
                )
                if targets is not None:
                    self._register_lora_hooks(active_device, targets)

                for component in self._components:
                    stack.callback(component.deactivate)
                    component.activate(active_device)

                self._teardown_stack = stack.pop_all()
        except BaseException:
            self._clear_loras()
            self._active_device = None
            raise

    def deactivate(self) -> None:
        stack = self._teardown_stack
        self._teardown_stack = None
        try:
            self._clear_active_lora_hooks()
            if stack is not None:
                stack.close()
        finally:
            self._clear_loras()
            self._active_device = None

    @contextlib.contextmanager
    def optimizer_step(self) -> Iterator[None]:
        """Context manager wrapping the optimizer-step boundary for
        managed trainable weights.

        On CUDA activation, non-streamed trainables are already active
        through :class:`PinnedComponent`, while streamed-component trainables
        are materialized on enter after force-evicting loaded blocks.
        On exit, updated trainable bytes are copied back to their pinned
        CPU storage. On CPU activation, this is a guarded no-op.

        ``param.grad`` is unaffected throughout. On CUDA, it lives on
        GPU during backward via PyTorch's native ``AccumulateGrad`` and
        is read+modified by the optimizer in place. ``optimizer.zero_grad()``,
        ``clip_grad_norm_``, AMP's ``GradScaler.unscale_`` and other
        grad-walking tools work as in vanilla PyTorch — they don't need
        to be inside this context.

        Typical loop::

            loss.backward()
            with offload.optimizer_step():
                optimizer.step()
            optimizer.zero_grad()
        """
        with contextlib.ExitStack() as stack:
            if self._pinned_component is not None:
                stack.enter_context(self._pinned_component.optimizer_step())
            for streamed_component in self._streamed_components:
                if streamed_component.has_trainables:
                    stack.enter_context(streamed_component.optimizer_step())
            yield

    @contextlib.contextmanager
    def gather_for_step(self) -> Iterator[None]:
        """Backward-compatible alias for :meth:`optimizer_step`.

        The public API names the boundary after the operation that
        requires all streamed trainable weight data to be materialized: the
        optimizer step.
        """
        with self.optimizer_step():
            yield

    @contextlib.contextmanager
    def use(self, device: torch.device | str) -> Iterator[nn.Module]:
        """Activate on ``device`` for the duration of the context."""
        self.activate(device)
        try:
            yield self.model
        finally:
            self.deactivate()

    # ----------------------------------------------------------- Internals

    @property
    def _has_streamed_blocks(self) -> bool:
        return any(
            _component_streams_tensor_state(component)
            for component in self._streamed_components
        )

    def _iter_streamed_block_groups(
        self,
    ) -> Iterator[tuple[StreamedComponent, list[nn.Module]]]:
        for streamed_component, blocks in zip(
            self._streamed_components,
            self._block_groups,
            strict=True,
        ):
            if _component_streams_tensor_state(streamed_component):
                yield streamed_component, blocks

    def _enforce_checkpointing_for_trainable_streaming(self) -> None:
        """Hard-guard: refuse to activate if a streamed_component manages
        trainable params and the configured checkpointing predicate
        returns ``False`` for any block in that streamed_component.

        Trainable streaming uses ``.data`` swap on the user's
        ``Parameter`` (preserves identity for autograd / optimizer
        state). Without checkpointing, autograd's version-counter
        check is **bypassed** — ``Tensor.set_`` (which
        ``param.data = ...`` invokes) repoints storage without
        bumping anything autograd sees. Result: silent gradient
        corruption rather than a loud error. Frozen streaming has
        the same precondition but a loud failure mode (saved-tensor
        version check raises), so it's a warning; trainable
        streaming is silent, so it's a hard error.

        Detection layers:

        - ``skip_checkpointing_check=True`` short-circuits — caller
          takes responsibility (typically used with manual call-site
          ``torch.utils.checkpoint.checkpoint`` wrapping, which is
          invisible from the module tree).
        - ``is_block_checkpointed: Callable[[nn.Module], bool]`` is
          the configurable per-block predicate. Default checks the
          HuggingFace per-block flag; non-HF frameworks (DeepSpeed,
          custom training loops) plug in their own detection.

        We deliberately do NOT walk module ancestors looking for the
        flag. Some HF model versions set ``gradient_checkpointing``
        on the root ``PreTrainedModel`` rather than each block; an
        ancestor walk would silence the guard for those — but a
        root-level flag does not prove that each block actually
        runs under checkpointing, so accepting it would re-introduce
        the silent-corruption failure mode. Users on those model
        families should either pass an ``is_block_checkpointed``
        predicate that knows the framework's conventions, or
        ``skip_checkpointing_check=True`` after manually verifying.
        """
        if not self._has_streamed_blocks:
            return
        if self._skip_checkpointing_check:
            return
        if not self._stream_trainable_weights:
            return
        if not self._model.training:
            return

        for streamed_component, blocks in self._iter_streamed_block_groups():
            if not streamed_component.has_trainables:
                continue
            for block in blocks:
                if not self._is_block_checkpointed(block):
                    raise RuntimeError(
                        "ModelOffloader: in-block trainable params "
                        "detected but the configured checkpointing "
                        "predicate (`is_block_checkpointed`) reports "
                        "False for at least one streamed block. "
                        "Trainable streaming uses .data swap, which "
                        "bypasses autograd's version-counter check "
                        "and silently corrupts gradients without "
                        "checkpointing — so this is a hard error, "
                        "not a warning.\n\n"
                        "Fix: call model.gradient_checkpointing_enable() "
                        "before constructing the ModelOffloader (HF models, "
                        "default detection); pass an "
                        "is_block_checkpointed predicate matching your "
                        "framework's conventions; or, if you wrap each "
                        "streamed block manually via "
                        "torch.utils.checkpoint.checkpoint at call "
                        "sites, pass skip_checkpointing_check=True to "
                        "suppress this guard (call-site wrapping is "
                        "invisible from the module tree)."
                    )

    def _warn_if_training_without_checkpointing(self) -> None:
        """Emit a one-time warning when training-shaped use is detected
        without configured checkpointing detection on streamed
        blocks.

        The default predicate only catches the HF flag
        (``module.gradient_checkpointing = True``, set by
        ``model.gradient_checkpointing_enable()``). Manual
        :func:`torch.utils.checkpoint.checkpoint` wrapping at call sites
        is invisible from the module tree, so callers using that style
        should pass ``skip_checkpointing_check=True``.
        """
        if not self._has_streamed_blocks:
            return
        if self._warned_about_checkpointing:
            return
        if self._skip_checkpointing_check:
            return
        if not self._model.training:
            return
        # Tighten the false-positive case for inference users who left
        # the model in train mode: only warn when at least one trainable
        # param actually exists.
        if not any(p.requires_grad for p in self._model.parameters()):
            return

        any_with = False
        any_without = False
        for _streamed_component, blocks in self._iter_streamed_block_groups():
            for block in blocks:
                if self._is_block_checkpointed(block):
                    any_with = True
                else:
                    any_without = True

        if any_with and any_without:
            logger.warning(
                "ModelOffloader: streamed blocks have inconsistent "
                "gradient_checkpointing flags. Backward through any "
                "non-checkpointed streamed block will raise on the first "
                "target reuse — call model.gradient_checkpointing_enable() "
                "to enable on every block."
            )
            self._warned_about_checkpointing = True
        elif not any_with:
            logger.warning(
                "ModelOffloader: model.training=True with trainable "
                "params, but no gradient_checkpointing flag is set on "
                "the streamed blocks. Training through streamed blocks "
                "requires checkpointing each block "
                "(model.gradient_checkpointing_enable() for HF models, "
                "or torch.utils.checkpoint.checkpoint() at call sites). "
                "Without it, loss.backward() will raise an in-place "
                "modification error from autograd's saved-tensor check. "
                "If you are wrapping blocks manually at call sites, "
                "ignore this warning (the wrap is invisible from the "
                "module tree)."
            )
            self._warned_about_checkpointing = True

# ---------------------------------------------------------------------------
# Module-private helpers (used only by ModelOffloader constructor)
# ---------------------------------------------------------------------------


def _normalize_layer_paths(layers_attr: str | Sequence[str] | None) -> list[str]:
    if layers_attr is None:
        return []
    layer_paths = [layers_attr] if isinstance(layers_attr, str) else list(layers_attr)
    if not layer_paths:
        raise ValueError("layers_attr must contain at least one path")
    return layer_paths


def _validate_constructor_mode(
    *,
    layer_paths: Sequence[str],
    blocks_to_swap: int | Sequence[int] | None,
    include_param_names: Iterable[str] | None,
    include_buffer_names: Iterable[str] | None,
) -> None:
    if layer_paths and blocks_to_swap is None:
        raise TypeError("ModelOffloader requires blocks_to_swap when layers_attr is set")
    if not layer_paths and blocks_to_swap is not None:
        raise ValueError("blocks_to_swap requires layers_attr")
    if layer_paths and (
        include_param_names is not None or include_buffer_names is not None
    ):
        raise ValueError(
            "include_param_names/include_buffer_names are only valid "
            "when layers_attr is omitted."
        )


def _build_streamed_component_stores(
    model: nn.Module,
    *,
    layer_paths: Sequence[str],
    blocks_to_swap: int | Sequence[int] | None,
    prefetch_count: int | Sequence[int],
    cyclic: bool,
    stream_trainable_weights: bool,
) -> tuple[tuple[StreamedComponentStore, ...], set[str], set[str]]:
    if not layer_paths:
        return (), set(), set()

    assert blocks_to_swap is not None
    swap_list = _broadcast(blocks_to_swap, len(layer_paths), "blocks_to_swap")
    pf_list = _broadcast(prefetch_count, len(layer_paths), "prefetch_count")
    streamed_component_stores: list[StreamedComponentStore] = []
    streamed_param_names: set[str] = set()
    streamed_buffer_names: set[str] = set()

    for i, layer_path in enumerate(layer_paths):
        store = StreamedComponentStore.from_module(
            model,
            layer_path=layer_path,
            blocks_to_swap=swap_list[i],
            prefetch_count=pf_list[i],
            cyclic=cyclic,
            stream_trainable_weights=stream_trainable_weights,
        )
        streamed_param_names.update(store.param_names)
        streamed_buffer_names.update(store.buffer_names)
        streamed_component_stores.append(store)

    return tuple(streamed_component_stores), streamed_param_names, streamed_buffer_names


def _build_pinned_component_store(
    model: nn.Module,
    *,
    streamed_param_names: set[str],
    streamed_buffer_names: set[str],
    include_param_names: Iterable[str] | None,
    include_buffer_names: Iterable[str] | None,
) -> PinnedComponentStore | None:
    pinned_param_names = parameter_names(model) - streamed_param_names
    pinned_buffer_names = buffer_names(model) - streamed_buffer_names
    if include_param_names is not None:
        pinned_param_names = set(include_param_names)
    if include_buffer_names is not None:
        pinned_buffer_names = set(include_buffer_names)
    if not pinned_param_names and not pinned_buffer_names:
        return None
    return PinnedComponentStore.from_module(
        model,
        include_param_names=pinned_param_names,
        include_buffer_names=pinned_buffer_names,
    )


def _compose_components(
    pinned_component: PinnedComponent | None,
    streamed_components: Sequence[StreamedComponent],
) -> list[ModelStrategyComponent]:
    components: list[ModelStrategyComponent] = []
    if pinned_component is not None:
        components.append(pinned_component)
    components.extend(streamed_components)
    if not components:
        raise ValueError(
            "ModelOffloader requires at least one parameter, registered "
            "buffer, or streamed block to manage."
        )
    return components


def _component_streams_tensor_state(component: StreamedComponent) -> bool:
    return bool(component.param_names) or any(component.streamed_buffer_names_by_block)


def _broadcast(value: int | Sequence[int], n: int, name: str) -> list[int]:
    if isinstance(value, int):
        return [value] * n
    out = list(value)
    if len(out) != n:
        raise ValueError(f"{name} length {len(out)} != layers_attr length {n}")
    return out


def _hf_block_has_checkpointing_flag(block: nn.Module) -> bool:
    """Default ``is_block_checkpointed`` predicate.

    Checks the HuggingFace per-block ``gradient_checkpointing``
    boolean attribute set by older ``model.gradient_checkpointing_enable()``
    implementations (and still used by many HF model families). Does
    NOT walk ancestors; conservative on purpose — see
    :meth:`ModelOffloader._enforce_checkpointing_for_trainable_streaming`
    for why.
    """
    return bool(getattr(block, "gradient_checkpointing", False))
