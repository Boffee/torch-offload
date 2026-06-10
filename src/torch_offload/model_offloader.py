"""Unified CUDA offload binding with optional LoRA application.

Supports whole-model pinned bulk offload or block streaming, with optional
per-weight LoRA application in both modes.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
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
class ModelOffloaderStore:
    """Reusable pinned backing storage for :class:`ModelOffloader`.

    Build once from a primary model, then bind to that model or to
    compatible model instances with :meth:`bind`.
    """

    model: nn.Module = field(repr=False, compare=False)
    pinned_component_store: PinnedComponentStore | None
    streamed_component_stores: tuple[StreamedComponentStore, ...]
    stream_trainable_weights: bool

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        blocks_attr: str | Sequence[str] | None = None,
        num_resident_blocks: int | None = None,
        num_prefetch_blocks: int = 2,
        cyclic: bool = False,
        stream_trainable_weights: bool = False,
    ) -> ModelOffloaderStore:
        blocks_paths = _normalize_blocks_paths(blocks_attr)
        _validate_store_config(
            blocks_paths=blocks_paths,
            num_resident_blocks=num_resident_blocks,
        )
        (
            streamed_component_stores,
            streamed_param_names,
            streamed_buffer_names,
        ) = _build_streamed_component_stores(
            model,
            blocks_paths=blocks_paths,
            num_resident_blocks=num_resident_blocks,
            num_prefetch_blocks=num_prefetch_blocks,
            cyclic=cyclic,
            stream_trainable_weights=stream_trainable_weights,
        )
        pinned_component_store = _build_pinned_component_store(
            model,
            streamed_param_names=streamed_param_names,
            streamed_buffer_names=streamed_buffer_names,
        )
        return cls(
            model=model,
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

    @property
    def has_trainables(self) -> bool:
        pinned_has_trainables = (
            self.pinned_component_store is not None
            and self.pinned_component_store.has_trainables
        )
        return pinned_has_trainables or any(
            store.has_trainables for store in self.streamed_component_stores
        )

    def bind(
        self,
        model: nn.Module,
        *,
        skip_checkpointing_check: bool = False,
        is_block_checkpointed: Callable[[nn.Module], bool] | None = None,
    ) -> ModelOffloader:
        """Bind this store's backing bytes to ``model``."""
        streamed_components = [
            store.bind(model)
            for store in self.streamed_component_stores
        ]
        pinned_component = (
            None
            if self.pinned_component_store is None
            else self.pinned_component_store.bind(model)
        )
        return ModelOffloader(
            model,
            pinned_component=pinned_component,
            streamed_components=streamed_components,
            stream_trainable_weights=self.stream_trainable_weights,
            skip_checkpointing_check=skip_checkpointing_check,
            is_block_checkpointed=is_block_checkpointed,
        )


__all__ = [
    "ModelOffloader",
    "ModelOffloaderStore",
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

    Instances are normally created by binding a
    :class:`ModelOffloaderStore` to a compatible model. Direct
    construction accepts already-bound components for low-level
    composition; it does not build stores or pin model state itself.

    When ``blocks_attr`` or ``num_resident_blocks`` is omitted
    (``num_resident_blocks=None`` disables streaming), CUDA activation
    bulk-copies every managed parameter and buffer to CUDA. When both
    are set, CUDA activation streams the selected block groups plus
    component-level movement for non-streamed state. CPU activation is
    pass-through over the pinned host-backed module state.

    Composes :class:`PinnedComponent` (non-streamed params and buffers)
    and one or more :class:`StreamedComponent`\\ s internally. LoRA requests
    are recorded via
    :meth:`set_loras` and applied on :meth:`activate`, where merge mode
    installs activation-scoped post-copy hooks so the merge fires
    immediately after each CPU->GPU weight copy. No separate merge
    binding is needed.

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
    GPU-resident while the offloader binding is active on CUDA, and must be
    updated inside :meth:`optimizer_step` so CUDA updates are copied
    back to the pinned CPU cache. CPU activation leaves them in the
    host-backed module state.

    Configure ``stream_trainable_weights=True`` on
    :meth:`ModelOffloaderStore.from_module` to stream in-block trainable
    parameter data through the CUDA block target pool. In that mode,
    CUDA :meth:`activate` *raises* during training if no
    ``gradient_checkpointing`` flag is detected. The failure mode here
    is **silent gradient corruption** rather than a loud error — the
    ``.data`` swap path used for trainable streaming bypasses autograd's
    version-counter check, so we hard-guard the precondition. Pass
    ``skip_checkpointing_check=True`` to :meth:`ModelOffloaderStore.bind`
    to suppress the check if you wrap blocks manually via
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
        The concrete model bound to the supplied components.
    pinned_component:
        Optional bound component for non-streamed parameter and buffer
        state.
    streamed_components:
        Bound streamed block-list components.
    stream_trainable_weights:
        Whether the streamed components include trainable parameter data.
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
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        pinned_component: PinnedComponent | None,
        streamed_components: Sequence[StreamedComponent] = (),
        stream_trainable_weights: bool,
        skip_checkpointing_check: bool,
        is_block_checkpointed: Callable[[nn.Module], bool] | None,
    ) -> None:
        self._model = model
        self._active_device: torch.device | None = None
        self._pinned_component = pinned_component
        self._streamed_components = list(streamed_components)
        _validate_components(pinned_component, self._streamed_components)
        if pinned_component is not None:
            self._instance = pinned_component._instance
        self._teardown_stack: contextlib.ExitStack | None = None
        self._lora_hook_handles: list[_RemovableHook] = []
        self._warned_about_checkpointing: bool = False
        self._stream_trainable_weights: bool = stream_trainable_weights
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
        loras: Sequence[LoRA],
        *,
        strengths: Sequence[float] | None = None,
        mode: LoraMode = "merge",
    ) -> None:
        """Record LoRAs for the next activation cycle.

        Must be called while deactivated. Attachments are cleared
        automatically on :meth:`deactivate`, so callers must call
        ``set_loras`` before each :meth:`activate` if they want
        LoRA-augmented inference.

        ``mode``:

        LoRA target keys must be canonical managed parameter names.
        Unknown targets raise during activation. PEFT ``.base_layer.``
        model parameter paths are canonicalized for lookup, so a LoRA
        target like ``"blocks.0.attn.weight"`` can match a managed model
        parameter named ``"blocks.0.attn.base_layer.weight"``.

        - ``"merge"`` (default): on CUDA activation, installs a
          post-copy hook per target. The hook applies
          :class:`LoRATransform` immediately after the base weight is
          copied to GPU, so it rides along with the streaming cycle.
          Requires CUDA activation and a base-weight adapter that
          supports dense in-place ``addmm_`` or dequantize/requantize
          plus ``copy_into`` merge.
        - ``"routed"``: on activation, registers a forward hook on each
          target's parent module. Forward becomes
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

        ``strengths`` defaults to ``1.0`` for each LoRA. Pass an empty
        LoRA sequence to clear all LoRAs (base-only forward).
        """
        if self._teardown_stack is not None:
            raise RuntimeError(
                "ModelOffloader.set_loras() requires the binding "
                "to be inactive. Call deactivate() first."
            )
        if mode not in ("merge", "routed"):
            raise ValueError(
                f"set_loras mode must be 'merge' or 'routed', got {mode!r}"
            )
        lora_list = list(loras)
        if strengths is None:
            strength_list = [1.0] * len(lora_list)
        else:
            if len(strengths) != len(lora_list):
                raise ValueError(
                    "strengths must have the same length as loras"
                )
            strength_list = [float(strength) for strength in strengths]
        for lora in lora_list:
            if not isinstance(lora, LoRA):
                raise TypeError(
                    "ModelOffloader.set_loras() expects LoRA instances"
                )
        self._loras = list(zip(lora_list, strength_list, strict=True))
        self._lora_mode = mode if lora_list else "merge"

    def _group_lora_factors_by_param_name(
        self, loras: Sequence[tuple[LoRA, float]],
    ) -> _LoraParamMap:
        per_param: _LoraParamMap = {}
        param_names = self.param_names
        canonical_param_names = {
            canonical_param_name(param_name): param_name
            for param_name in param_names
        }
        for lora, strength in loras:
            for target_key, (a, b) in lora.targets.items():
                param_name = canonical_param_names.get(target_key)
                if param_name is None:
                    sample_index = sorted(canonical_param_names)[:3]
                    raise ValueError(
                        f"LoRA target {target_key!r} is not managed by "
                        "this ModelOffloader. LoRA target keys must use "
                        "canonical model parameter names. Sample managed "
                        f"keys: {sample_index} ..."
                    )
                per_param.setdefault(param_name, []).append(
                    (a, b, strength)
                )

        return per_param

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
    def active_device(self) -> torch.device | None:
        """Currently active device, or ``None`` when inactive."""
        return self._active_device

    @property
    def param_names(self) -> frozenset[str]:
        """Parameter names managed by this offloader."""
        names: set[str] = set()
        if self._pinned_component is not None:
            names.update(self._pinned_component.param_names)
        for streamed_component in self._streamed_components:
            names.update(streamed_component.param_names)
        return frozenset(names)

    @property
    def buffer_names(self) -> frozenset[str]:
        """Buffer names managed by this offloader."""
        names: set[str] = set()
        if self._pinned_component is not None:
            names.update(self._pinned_component.buffer_names)
        for streamed_component in self._streamed_components:
            names.update(streamed_component.buffer_names)
        return frozenset(names)

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        if device is not None:
            return canonical_device(device)
        raise ValueError(
            "ModelOffloader.activate() requires a device; pass "
            "activate(device) or use this binding through "
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

                for component in self._iter_components():
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

    def _iter_components(self) -> Iterator[ModelStrategyComponent]:
        if self._pinned_component is not None:
            yield self._pinned_component
        yield from self._streamed_components

    @property
    def _has_streamed_blocks(self) -> bool:
        return any(
            _component_streams_tensor_state(component)
            for component in self._streamed_components
        )

    def _iter_streamed_block_groups(
        self,
    ) -> Iterator[tuple[StreamedComponent, tuple[nn.Module, ...]]]:
        for streamed_component in self._streamed_components:
            if _component_streams_tensor_state(streamed_component):
                yield streamed_component, streamed_component.blocks

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
                        "before binding or activating the ModelOffloader "
                        "(HF models, "
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
# Module-private helpers
# ---------------------------------------------------------------------------


def _normalize_blocks_paths(blocks_attr: str | Sequence[str] | None) -> list[str]:
    if blocks_attr is None:
        return []
    blocks_paths = [blocks_attr] if isinstance(blocks_attr, str) else list(blocks_attr)
    if not blocks_paths:
        raise ValueError("blocks_attr must contain at least one path")
    return blocks_paths


def _validate_store_config(
    *,
    blocks_paths: Sequence[str],
    num_resident_blocks: int | None,
) -> None:
    if not blocks_paths and num_resident_blocks is not None:
        raise ValueError("num_resident_blocks requires blocks_attr")


def _build_streamed_component_stores(
    model: nn.Module,
    *,
    blocks_paths: Sequence[str],
    num_resident_blocks: int | None,
    num_prefetch_blocks: int,
    cyclic: bool,
    stream_trainable_weights: bool,
) -> tuple[tuple[StreamedComponentStore, ...], set[str], set[str]]:
    # num_resident_blocks=None disables streaming even when
    # blocks_attr is set — the blocks are then managed by the
    # whole-model pinned component like any other module state.
    if not blocks_paths or num_resident_blocks is None:
        return (), set(), set()

    streamed_component_stores: list[StreamedComponentStore] = []
    streamed_param_names: set[str] = set()
    streamed_buffer_names: set[str] = set()

    for blocks_path in blocks_paths:
        store = StreamedComponentStore.from_module(
            model,
            blocks_path=blocks_path,
            num_resident_blocks=num_resident_blocks,
            num_prefetch_blocks=num_prefetch_blocks,
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
) -> PinnedComponentStore | None:
    pinned_param_names = parameter_names(model) - streamed_param_names
    pinned_buffer_names = buffer_names(model) - streamed_buffer_names
    if not pinned_param_names and not pinned_buffer_names:
        return None
    return PinnedComponentStore.from_module(
        model,
        include_param_names=pinned_param_names,
        include_buffer_names=pinned_buffer_names,
    )


def _validate_components(
    pinned_component: PinnedComponent | None,
    streamed_components: Sequence[StreamedComponent],
) -> None:
    if pinned_component is None and not streamed_components:
        raise ValueError(
            "ModelOffloader requires at least one parameter, registered "
            "buffer, or streamed block to manage."
        )


def _component_streams_tensor_state(component: StreamedComponent) -> bool:
    return bool(component.param_names) or any(component.streamed_buffer_names_by_block)


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
