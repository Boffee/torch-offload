"""Unified CUDA offload binding with optional LoRA application.

Supports whole-model pinned bulk offload or block streaming, with optional
per-weight LoRA application in both modes.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import torch
from torch import nn

from ._devices import canonical_device
from .composite_component import CompositeComponent, CompositeComponentStore
from .lora import LoRA, LoRARouteHandle, LoRATransform, ScaledLoRAFactor
from .module_names import (
    canonical_param_name,
    resolve_parent_leaf,
)
from .tensor_adapter_registry import param_representation, select_adapter

LoraMode = Literal["merge", "routed"]
_LoraParamMap = dict[str, list[ScaledLoRAFactor]]


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
    composite_store: CompositeComponentStore

    @classmethod
    def from_module(
        cls,
        model: nn.Module,
        *,
        blocks_attr: list[str] = [],  # noqa: B006  (read-only; never mutated)
        stream_trainable_weights: bool = False,
    ) -> ModelOffloaderStore:
        composite_store = CompositeComponentStore.from_module(
            model,
            blocks_attr=blocks_attr,
            stream_trainable_weights=stream_trainable_weights,
        )
        return cls(model=model, composite_store=composite_store)

    @property
    def cache_bytes(self) -> int:
        return self.composite_store.cache_bytes

    @property
    def has_trainables(self) -> bool:
        return self.composite_store.has_trainables

    def bind(self, model: nn.Module) -> ModelOffloader:
        """Bind this store's backing bytes to ``model``."""
        return ModelOffloader(model, composite=self.composite_store.bind(model))


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
    - ``select_adapter(...).compute_dtype(...)`` on the weight's
      representation — plain tensors return ``weight.dtype``;
      structured/quantized wrappers can return their logical matmul dtype
      instead of packed storage dtype.
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
    representation = param_representation(weight)
    return select_adapter(representation).compute_dtype(representation)


class ModelOffloader:
    """Move a whole model or streamed block groups between pinned CPU and
    CUDA, with optional LoRA merge and trainable-parameter support.

    Instances are normally created by binding a
    :class:`ModelOffloaderStore` to a compatible model. Direct
    construction accepts already-bound components for low-level
    composition; it does not build stores or pin model state itself.

    When ``blocks_attr`` is omitted, CUDA activation bulk-copies every
    managed parameter and buffer to CUDA. When it is set, CUDA activation
    streams the selected block groups plus component-level movement for
    non-streamed state. CPU activation is pass-through over the pinned
    host-backed module state.

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
    outside that window is then safe. Ensuring each streamed block that
    participates in training is checkpointed is the caller's
    responsibility — there is no auto-detection or guard.

    By default, trainable params are not streamed through the block
    residency pool. They are managed by :class:`PinnedComponent`, stay
    GPU-resident while the offloader binding is active on CUDA, and must be
    updated inside :meth:`optimizer_step` so CUDA updates are copied
    back to the pinned CPU cache. CPU activation leaves them in the
    host-backed module state.

    Configure ``stream_trainable_weights=True`` on
    :meth:`ModelOffloaderStore.from_module` to stream in-block trainable
    parameter data through the CUDA block target pool. In that mode,
    :meth:`optimizer_step` is the optimizer boundary: it materializes
    streamed trainable ``.data`` on GPU while an arbitrary PyTorch
    optimizer updates it, then copies the updated data back to pinned
    CPU. CPU activation makes :meth:`optimizer_step` a guarded no-op.
    Gradients are not streamed; PyTorch owns ``param.grad`` normally.

    Parameters
    ----------
    model:
        The concrete model bound to the supplied composite.
    composite:
        Bound :class:`CompositeComponent` owning the model's pinned
        and streamed offload components.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        composite: CompositeComponent,
    ) -> None:
        self._model = model
        self._active_device: torch.device | None = None
        self._composite = composite
        self._lora_hook_handles: list[_RemovableHook] = []
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
        if self._active_device is not None:
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
            for target_key, factor in lora.targets.items():
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
                    factor.scaled(strength)
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

        for param_name, factors in targets.items():
            transform = LoRATransform(factors)
            handle = self._register_post_copy_hook(
                param_name, transform.apply,
            )
            self._lora_hook_handles.append(handle)

    def _register_routed_lora_hooks(
        self,
        active_device: torch.device,
        targets: _LoraParamMap,
    ) -> None:
        for param_name, factors in targets.items():
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
                parent, factors, active_device,
                dtype=_routed_factor_dtype(parent),
            )
            self._lora_hook_handles.append(handle)

    def _register_post_copy_hook(
        self,
        param_name: str,
        hook: Callable[[nn.Parameter], None],
    ) -> _RemovableHook:
        return self._composite.register_post_copy_hook(param_name, hook)

    def register_post_copy_hook(
        self,
        param_name: str,
        hook: Callable[[nn.Parameter], None],
    ) -> _RemovableHook:
        """Register a hook after the owning component copies ``param_name``."""
        return self._register_post_copy_hook(param_name, hook)

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
        return self._composite.param_names

    @property
    def buffer_names(self) -> frozenset[str]:
        """Buffer names managed by this offloader."""
        return self._composite.buffer_names

    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        if device is not None:
            return canonical_device(device)
        raise ValueError(
            "ModelOffloader.activate() requires a device; pass "
            "activate(device) or use this binding through "
            "ModelCache.use(..., device=...)"
        )

    def activate(
        self, device: torch.device | str | None = None, **kwargs: object,
    ) -> None:
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
        self._active_device = active_device
        try:
            # Register LoRA hooks (incl. merge post-copy hooks installed on the
            # components) BEFORE activating, so they fire during component loads.
            targets = (
                self._group_lora_factors_by_param_name(self._loras)
                if self._loras
                else None
            )
            if targets is not None:
                self._register_lora_hooks(active_device, targets)
            # The composite self-cleans its components if activation fails midway.
            self._composite.activate(active_device, **kwargs)
        except BaseException:
            self._clear_loras()
            self._active_device = None
            raise

    def deactivate(self) -> None:
        try:
            self._clear_active_lora_hooks()
            self._composite.deactivate()
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

        This context steps on the *GPU* for speed. To run the optimizer on
        *CPU* instead — keeping its state on the host — call
        ``optimizer.step()`` **outside** ``use()`` (deactivated) without this
        context. On ``deactivate()`` every managed trainable has its ``.data``
        restored to pinned CPU storage *and* its ``.grad`` moved to CPU
        (:class:`PinnedComponent` and :class:`StreamedComponent` alike), so
        the step runs on CPU and the in-place update is streamed to GPU on the
        next forward. Keep such trainables in fp32 so the update is a correct
        master-weight update::

            with offload.use("cuda"):
                loss = model(x); loss.backward()
            optimizer.step()        # runs on CPU; states stay on host
            optimizer.zero_grad()
        """
        with self._composite.optimizer_step():
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
    def use(
        self, device: torch.device | str, **kwargs: object,
    ) -> Iterator[nn.Module]:
        """Activate on ``device`` for the duration of the context."""
        self.activate(device, **kwargs)
        try:
            yield self.model
        finally:
            self.deactivate()
