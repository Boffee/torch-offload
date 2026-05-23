"""Unified block-streaming strategy with optional LoRA application.

Composes block streaming, non-streamed pinning, and optional per-weight
LoRA application into a single :class:`ModelOffloader` class.

Also provides :func:`detect_streaming_region_ties`
(construction-time validation), used internally by
:class:`ModelOffloader` and exported for direct use / testing.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import nn

from ._devices import canonical_device
from .lora import LoRA, LoRARouteHandle, LoRATransform
from .pinned_weights import PinnedWeights
from .protocols import ModelStrategyComponent, SlotKey
from .slots import (
    buffer_storage_key,
    canonical_param_name,
    iter_buffer_slots,
    iter_param_slots,
    param_storage_key,
    walk_attr_path,
)
from .streamed_weights import StreamedWeights
from .tensor_adapter_factory import select_adapter

logger = logging.getLogger(__name__)

LoraMode = Literal["merge", "routed"]
_LoraFactorRef = tuple[torch.Tensor, torch.Tensor, float]
_LoraTargetMap = dict[str, list[_LoraFactorRef]]


class _RemovableHook(Protocol):
    def remove(self) -> None:
        ...


__all__ = [
    "ModelOffloader",
    "detect_streaming_region_ties",
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
    """Stream transformer blocks between pinned CPU and CUDA with
    optional LoRA merge and trainable-parameter support.

    CUDA activation uses block streaming plus component-level device
    movement. CPU activation is pass-through over the pinned
    host-backed module slots.

    Composes :class:`PinnedWeights` (non-streamed params and buffers)
    and one or more :class:`StreamedWeights`\\ s internally. LoRA requests
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
    been modified by an inplace operation`` on the first slot reuse.

    Why: autograd saves a reference to each ``Linear``'s weight
    tensor at forward time and records its version counter. Streaming
    is a sequence of in-place ``copy_`` writes into a fixed pool of
    GPU slot tensors — every load bumps the slot tensor's version,
    invalidating any previously-saved reference into that slot.
    Checkpointing makes each block's internal forward run under
    ``no_grad`` (no internal tensors saved); when backward arrives,
    PyTorch re-runs the block's forward with grad enabled, building
    a fresh autograd graph whose saved references only live within
    that one block's recompute-then-backward window. Slot reuse
    outside that window is then safe.

    For **frozen-only** streamed blocks (training touches only
    out-of-block trainables), CUDA :meth:`activate` emits a one-time
    warning if no HuggingFace ``gradient_checkpointing`` flag is
    detected — the failure mode without checkpointing is a loud
    ``RuntimeError`` from autograd's saved-tensor check, so a warning
    suffices.

    By default, trainable params are not streamed through the block
    residency pool. They are managed by :class:`PinnedWeights`, stay
    GPU-resident while the offloader is active on CUDA, and must be
    updated inside :meth:`optimizer_step` so CUDA updates are copied
    back to the pinned CPU cache. CPU activation leaves them in the
    host-backed module state.

    Pass ``stream_trainable_weights=True`` to stream in-block
    trainable parameter data through the CUDA block slot pool. In that
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
        The model containing the block list(s). Managed slots may start
        on CPU or CUDA; construction clones them directly into pinned
        CPU storage before activation.
    layers_attr:
        Dotted attribute path(s) to ``nn.ModuleList`` block list(s).
        Single string or sequence. For PEFT-wrapped models, include
        the PEFT prefix (e.g. ``"base_model.model.transformer_blocks"``).
    blocks_to_swap:
        Per-group count of blocks to keep on CPU. Single int (broadcast
        to all groups) or one int per group.
    prefetch_count:
        Per-group prefetch depth. Same broadcasting as *blocks_to_swap*.
    cyclic:
        Default ``False``. Forwarded to every :class:`StreamedWeights`.
        Set ``True`` for inference loops that iterate the model
        repeatedly (diffusion denoising, multi-step decoders); the
        prefetcher then treats end-of-iteration as wraparound and
        keeps streaming the next iteration's leading blocks. Leave
        ``False`` for single-shot inference or training.
    stream_trainable_weights:
        Default ``False`` skips trainable params in block streaming and
        manages them with :class:`PinnedWeights`. ``True`` streams
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
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        layers_attr: str | Sequence[str],
        blocks_to_swap: int | Sequence[int],
        prefetch_count: int | Sequence[int] = 2,
        cyclic: bool = False,
        stream_trainable_weights: bool = False,
        skip_checkpointing_check: bool = False,
        is_block_checkpointed: Callable[[nn.Module], bool] | None = None,
    ) -> None:
        layer_paths: list[str] = (
            [layers_attr] if isinstance(layers_attr, str) else list(layers_attr)
        )
        if not layer_paths:
            raise ValueError("layers_attr must contain at least one path")

        n = len(layer_paths)
        swap_list = _broadcast(blocks_to_swap, n, "blocks_to_swap")
        pf_list = _broadcast(prefetch_count, n, "prefetch_count")

        block_groups: list[list[nn.Module]] = [
            list(_resolve_layers_attr(model, p)) for p in layer_paths
        ]
        for i, blocks in enumerate(block_groups):
            if not blocks:
                raise ValueError(
                    f"layers_attr[{i}] = {layer_paths[i]!r} resolved to empty list"
                )

        _validate_block_groups_disjoint(block_groups, layer_paths)
        _validate_streamed_block_names_are_exclusive(
            model,
            block_groups,
            layer_paths,
            stream_trainables=stream_trainable_weights,
        )
        detect_streaming_region_ties(
            model, block_groups, stream_trainables=stream_trainable_weights,
        )

        # By default, streamers skip trainables and PinnedWeights keeps
        # them GPU-resident while active. With stream_trainable_weights=True,
        # streamers handle in-block params of both kinds: frozen via
        # parameter replacement and trainable via identity-preserving
        # ``.data`` swap.
        # Gradients are NOT streamed — they live on GPU during backward via
        # PyTorch's native ``AccumulateGrad`` mechanism.
        streamers: list[StreamedWeights] = []
        streamed_param_names: set[str] = set()
        streamed_buffer_names: set[str] = set()
        for i, blocks in enumerate(block_groups):
            stream_param_names, stream_buffer_names = _streamed_names_for_blocks(
                blocks,
                stream_trainables=stream_trainable_weights,
            )
            streamed_param_names.update(
                _full_block_names(layer_paths[i], len(blocks), stream_param_names)
            )
            streamed_buffer_names.update(
                _full_block_names(layer_paths[i], len(blocks), stream_buffer_names)
            )
            streamers.append(
                StreamedWeights(
                    blocks=blocks,
                    blocks_to_swap=swap_list[i],
                    prefetch_count=pf_list[i],
                    cyclic=cyclic,
                    name=layer_paths[i],
                    stream_param_names=stream_param_names,
                    stream_buffer_names=stream_buffer_names,
                )
            )

        # PinnedWeights manages every non-streamed param and buffer,
        # including trainables. In default mode that includes in-block
        # trainables because streamers skipped them above.
        pinned_param_names = _all_param_names(model) - streamed_param_names
        pinned_buffer_names = _all_buffer_names(model) - streamed_buffer_names
        pinned_weights: PinnedWeights | None = None
        if pinned_param_names or pinned_buffer_names:
            pinned_weights = PinnedWeights(
                model,
                include_param_names=pinned_param_names,
                include_buffer_names=pinned_buffer_names,
            )

        components: list[ModelStrategyComponent] = []
        if pinned_weights is not None:
            components.append(pinned_weights)
        components.extend(streamers)

        self._model = model
        self._active_device: torch.device | None = None
        self._components = components
        self._pinned_weights = pinned_weights
        self._streamers = streamers
        self._teardown_stack: contextlib.ExitStack | None = None

        (
            self._target_to_param_name,
            self._target_to_parent,
            self._target_to_component,
        ) = self._build_target_index(
            streamers,
            layer_paths,
            block_groups,
            pinned_weights,
        )
        self._lora_hook_handles: list[_RemovableHook] = []
        self._block_groups: list[list[nn.Module]] = block_groups
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
                "ModelOffloader.set_loras() requires the offloader "
                "to be inactive. Call deactivate() first."
            )
        if mode not in ("merge", "routed"):
            raise ValueError(
                f"set_loras mode must be 'merge' or 'routed', got {mode!r}"
            )
        configured_loras = list(loras)
        self._loras = configured_loras
        self._lora_mode = mode if configured_loras else "merge"

    def _group_loras_by_target(
        self, loras: Sequence[tuple[LoRA, float]],
    ) -> _LoraTargetMap:
        per_target: _LoraTargetMap = {}
        total_targets = 0
        matched_targets = 0
        for lora, strength in loras:
            for target_key, (a, b) in lora.targets.items():
                total_targets += 1
                if target_key not in self._target_to_param_name:
                    continue
                matched_targets += 1
                per_target.setdefault(target_key, []).append(
                    (a, b, strength)
                )

        if matched_targets < total_targets:
            sample_lora = sorted(next(iter(loras))[0].targets)[:3]
            sample_index = sorted(self._target_to_param_name)[:3]
            logger.warning(
                "set_loras matched %d/%d targets. "
                "Sample LoRA keys: %s ... Sample index keys: %s ...",
                matched_targets, total_targets, sample_lora, sample_index,
            )
        else:
            logger.debug("set_loras matched %d/%d targets", matched_targets, total_targets)

        return per_target

    def _register_lora_hooks(
        self, active_device: torch.device, targets: _LoraTargetMap,
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
        self, active_device: torch.device, targets: _LoraTargetMap,
    ) -> None:
        if active_device.type != "cuda":
            raise ValueError(
                "ModelOffloader merge mode requires CUDA activation; "
                f"got {active_device}. Use set_loras(..., mode='routed') "
                "for CPU activation."
            )

        for target_key, refs in targets.items():
            param_name = self._target_to_param_name[target_key]
            component = self._target_to_component[target_key]
            transform = LoRATransform(refs)
            handle = component.register_post_copy_hook(
                param_name, transform.apply,
            )
            self._lora_hook_handles.append(handle)

    def _register_routed_lora_hooks(
        self,
        active_device: torch.device,
        targets: _LoraTargetMap,
    ) -> None:
        for target_key, refs in targets.items():
            parent = self._target_to_parent[target_key]
            if not isinstance(parent, nn.Linear):
                raise ValueError(
                    f"Routed LoRA mode requires nn.Linear targets; "
                    f"target {target_key!r} has parent module of "
                    f"type {type(parent).__name__}. Use mode='merge' "
                    f"for non-Linear targets, or wrap the model with "
                    f"PEFT for richer per-type routing."
                )
            handle = LoRARouteHandle(
                parent, refs, active_device,
                dtype=_routed_factor_dtype(parent),
            )
            self._lora_hook_handles.append(handle)

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
                    self._group_loras_by_target(self._loras)
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
        through :class:`PinnedWeights`, while streamer-managed trainables
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
            with offloader.optimizer_step():
                optimizer.step()
            optimizer.zero_grad()
        """
        if self._active_device is not None and self._active_device.type == "cpu":
            yield
            return
        with contextlib.ExitStack() as stack:
            if self._pinned_weights is not None:
                stack.enter_context(self._pinned_weights.optimizer_step())
            for streamer in self._streamers:
                if streamer.has_trainables:
                    stack.enter_context(streamer.optimizer_step())
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

    def _enforce_checkpointing_for_trainable_streaming(self) -> None:
        """Hard-guard: refuse to activate if a streamer manages
        trainable params and the configured checkpointing predicate
        returns ``False`` for any block in that streamer.

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
        if self._skip_checkpointing_check:
            return
        if not self._stream_trainable_weights:
            return
        if not self._model.training:
            return

        for streamer, blocks in zip(self._streamers, self._block_groups, strict=True):
            if not streamer.has_trainables:
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
                        "before constructing the offloader (HF models, "
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
        for blocks in self._block_groups:
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
                "slot reuse — call model.gradient_checkpointing_enable() "
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

    @staticmethod
    def _build_target_index(
        streamers: list[StreamedWeights],
        layer_paths: list[str],
        block_groups: list[list[nn.Module]],
        pinned_weights: PinnedWeights | None,
    ) -> tuple[
        dict[str, str],
        dict[str, nn.Module],
        dict[str, PinnedWeights | StreamedWeights],
    ]:
        """Map canonical param names to their exact runtime param name
        and to the exact parent module where the named param is installed.

        The param-name map drives merge-mode LoRA post-copy hooks, the
        component map identifies which component owns the copy loop, and
        the parent map drives routed-mode LoRA forward hooks.

        Keys are normalized to strip PEFT's ``.base_layer.`` segments
        so that LoRA state-dict keys (which use the original model
        names) match regardless of whether the model is PEFT-wrapped.

        Routed LoRA is name-centric: it hooks the parent module for the
        exact matched name. Shared parameter storage alone does not make
        routed mode ambiguous because routed mode does not mutate the
        parameter bytes.
        """
        param_names: dict[str, str] = {}
        parent_by_key: dict[str, nn.Module] = {}
        components: dict[str, PinnedWeights | StreamedWeights] = {}

        for streamer, layer_path, blocks in zip(
            streamers,
            layer_paths,
            block_groups,
            strict=True,
        ):
            for block_idx, local_names in enumerate(
                streamer.streamed_param_names_by_block
            ):
                block = blocks[block_idx]
                for local_name in local_names:
                    parent = _resolve_param_parent(block, local_name)
                    full_name = f"{layer_path}.{block_idx}.{local_name}"
                    key = canonical_param_name(full_name)
                    param_names[key] = full_name
                    parent_by_key[key] = parent
                    components[key] = streamer

        if pinned_weights is not None:
            for name in pinned_weights.param_names:
                key = canonical_param_name(name)
                param_names[key] = name
                parent_by_key[key] = _resolve_param_parent(
                    pinned_weights.model, name,
                )
                components[key] = pinned_weights

        return param_names, parent_by_key, components


# ---------------------------------------------------------------------------
# Module-private helpers (used only by ModelOffloader constructor)
# ---------------------------------------------------------------------------


def _streamed_names_for_blocks(
    blocks: Sequence[nn.Module],
    *,
    stream_trainables: bool,
) -> tuple[set[str], set[str]]:
    param_names = _block_param_names(blocks[0], stream_trainables=stream_trainables)
    buffer_names = _block_buffer_names(blocks[0])
    for i, block in enumerate(blocks[1:], start=1):
        if (
            _block_param_names(block, stream_trainables=stream_trainables)
            != param_names
            or _block_buffer_names(block) != buffer_names
        ):
            raise ValueError(
                f"Block {i} selected names differ from block 0. All blocks "
                "in a StreamedWeights group must select the same parameter "
                "and buffer names."
            )
    return param_names, buffer_names


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


def _full_block_names(
    layer_path: str,
    block_count: int,
    local_names: set[str],
) -> set[str]:
    return {
        f"{layer_path}.{block_idx}.{local_name}"
        for block_idx in range(block_count)
        for local_name in local_names
    }


def _all_param_names(model: nn.Module) -> set[str]:
    return {
        name
        for name, _param in model.named_parameters(remove_duplicate=False)
    }


def _all_buffer_names(model: nn.Module) -> set[str]:
    return {
        name
        for name, _buffer in model.named_buffers(remove_duplicate=False)
    }


def _resolve_param_parent(block: nn.Module, name: str) -> nn.Module:
    parent_path, sep, _leaf = name.rpartition(".")
    if not sep:
        return block
    parent = walk_attr_path(block, parent_path)
    if not isinstance(parent, nn.Module):
        raise TypeError(
            f"Path {parent_path!r} resolved to {type(parent).__name__}, "
            "expected nn.Module."
        )
    return parent


def _resolve_layers_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    obj = walk_attr_path(module, dotted_path)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(
            f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}"
        )
    return obj


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


# ---------------------------------------------------------------------------
# Composer-level invariants
# ---------------------------------------------------------------------------


def _validate_block_groups_disjoint(
    block_groups: Sequence[Sequence[nn.Module]], layer_paths: Sequence[str]
) -> None:
    """Reject configurations where any :class:`SlotKey` is owned
    by more than one block entry.

    Each ``StreamedWeights`` instance assumes it is the sole owner of
    every ``(parent_module, leaf, kind)`` slot inside its blocks.
    Overlapping ownership comes in three shapes:

    - **Same block module listed in two groups** (or twice in one):
      multiple block entries would pin and slot-replace the same
      ``nn.Module``, and per-block ``.data`` swap during streaming
      would conflict.
    - **Parent/child blocks across groups** (group A has a parent
      module, group B has one of its children): different
      ``id(parent)`` block instances, but their per-leaf
      ``SlotKey`` values coincide on the child's slots.
    - **Within a single group, the same nn.Module appearing twice**:
      the streamer would create per-index pinned clones for the same
      slots and swap ``.data`` redundantly.

    This validation is the precondition for
    :func:`detect_streaming_region_ties` and the per-block instance
    construction inside :class:`StreamedWeights` to be unambiguous; it lives
    at the composer because it is about how the caller composed
    ``layers_attr`` paths, not about parameter aliasing.
    """
    slot_owner: dict[SlotKey, str] = {}
    duplicates: dict[SlotKey, list[str]] = {}
    for group_idx, blocks in enumerate(block_groups):
        for block_idx, layer in enumerate(blocks):
            label = f"{layer_paths[group_idx]}[{block_idx}]"
            slots: set[SlotKey] = set()
            for s in iter_param_slots(layer):
                slots.add(s.key)
            for s in iter_buffer_slots(layer):
                slots.add(s.key)
            for slot in slots:
                if slot in slot_owner:
                    duplicates.setdefault(slot, [slot_owner[slot]]).append(label)
                else:
                    slot_owner[slot] = label
    if duplicates:
        sample = next(iter(duplicates.items()))
        raise ValueError(
            f"Streamer block groups must own disjoint module slots, "
            f"but {len(duplicates)} slot(s) are claimed by multiple "
            f"block entries. Example: leaf {sample[0].leaf!r} is "
            f"owned by {sample[1]}. Likely causes: the same block is "
            f"listed in two `layers_attr` paths, a path resolves to a "
            f"module that contains another resolved path, or a block "
            f"appears twice in a single `nn.ModuleList`."
        )


def _validate_streamed_block_names_are_exclusive(
    model: nn.Module,
    block_groups: Sequence[Sequence[nn.Module]],
    layer_paths: Sequence[str],
    *,
    stream_trainables: bool,
) -> None:
    """Reject params/buffers inside streamed blocks that have extra model names."""
    expected_names_by_slot: dict[SlotKey, set[str]] = {}
    for group_idx, blocks in enumerate(block_groups):
        layer_path = layer_paths[group_idx]
        for block_idx, block in enumerate(blocks):
            prefix = f"{layer_path}.{block_idx}"
            for slot in iter_param_slots(block):
                if not stream_trainables and slot.get().requires_grad:
                    continue
                expected_names_by_slot.setdefault(slot.key, set()).add(
                    f"{prefix}.{slot.name}"
                )
            for slot in iter_buffer_slots(block):
                expected_names_by_slot.setdefault(slot.key, set()).add(
                    f"{prefix}.{slot.name}"
                )

    for slot in iter_param_slots(model):
        expected_names = expected_names_by_slot.get(slot.key)
        if expected_names is not None and slot.name not in expected_names:
            _raise_non_exclusive_streamed_block_name(slot.name, expected_names)
    for slot in iter_buffer_slots(model):
        expected_names = expected_names_by_slot.get(slot.key)
        if expected_names is not None and slot.name not in expected_names:
            _raise_non_exclusive_streamed_block_name(slot.name, expected_names)


def _raise_non_exclusive_streamed_block_name(
    name: str,
    expected_names: set[str],
) -> None:
    raise ValueError(
        "ModelOffloader requires streamed block names to be exclusive. "
        f"Slot {name!r} is inside a configured streamed block but is also "
        f"reachable outside its expected streamed name(s) "
        f"{sorted(expected_names)!r}. Remove the extra module reference or "
        "use whole-model PinnedWeights."
    )


# ---------------------------------------------------------------------------
# Cross-region tied-weight detection
# ---------------------------------------------------------------------------


_NON_BLOCK_REGION = "non_block"
_BLOCK_REGION_PREFIX = "block:"


@dataclass(frozen=True, slots=True)
class _ParamStorageMember:
    region: str
    name: str
    requires_grad: bool


@dataclass(frozen=True, slots=True)
class _BufferStorageMember:
    region: str
    name: str


def detect_streaming_region_ties(
    model: nn.Module,
    block_groups: Sequence[Sequence[nn.Module]],
    *,
    stream_trainables: bool = False,
) -> None:
    """Raise if tied storage is unsafe for block streaming.

    Each entry in ``block_groups`` is one block list, and each block in
    those lists is treated as a separate streamed region. Everything
    else in ``model`` is the "non_block" region.

    Unsupported configurations:

    - **Frozen cross-region parameter ties** (block<->block,
      block<->non-block): the per-region pinning regimes can't
      coordinate to share storage.
    - **Mixed frozen/trainable ties** anywhere — across regions OR
      within a single region. The frozen side gets pinned and slot-
      replaced while the trainable side is moved via storage swap on
      activate; the two mechanisms cannot share a tied storage
      without breaking aliasing invariants silently.
    - **Streamed trainable ties across regions** when
      ``stream_trainables=True``. Each streamed region owns independent
      pinned storage and cannot coordinate optimizer-step copy-back for
      shared trainable storage. Intra-region trainable ties are safe:
      pinned module stores deduplicate them by storage and install a
      single active target.
    - **Cross-region buffer ties**. Per-block buffer clones and
      composed non-block pinning cannot coordinate to preserve the
      alias.
    Non-block-internal all-frozen ties (the standard ``tie_weights()``
    embed<->head pattern) are handled correctly by
    :class:`PinnedWeights`'s storage-key dedup and are NOT rejected
    here.
    """
    param_slot_regions = _param_slot_regions(block_groups)
    for members in _param_storage_groups(model, param_slot_regions).values():
        _validate_param_storage_group(
            members,
            stream_trainables=stream_trainables,
        )

    buffer_slot_regions = _buffer_slot_regions(block_groups)
    for members in _buffer_storage_groups(model, buffer_slot_regions).values():
        _validate_buffer_storage_group(members)


def _block_region(group_idx: int, block_idx: int) -> str:
    return f"{_BLOCK_REGION_PREFIX}{group_idx}:{block_idx}"


def _param_slot_regions(
    block_groups: Sequence[Sequence[nn.Module]],
) -> dict[SlotKey, str]:
    """Map slots inside streamed blocks to their owning block region.

    Slot-level ownership distinguishes "this slot lives in block X"
    from "this slot's Parameter is also referenced from block Y" — the
    latter is what cross-region same-Parameter aliasing produces. A
    Parameter-id-only map would silently collapse both slots to the
    first block's region.
    """
    regions: dict[SlotKey, str] = {}
    for group_idx, blocks in enumerate(block_groups):
        for block_idx, layer in enumerate(blocks):
            region = _block_region(group_idx, block_idx)
            for slot in iter_param_slots(layer):
                regions[slot.key] = region
    return regions


def _buffer_slot_regions(
    block_groups: Sequence[Sequence[nn.Module]],
) -> dict[SlotKey, set[str]]:
    regions: dict[SlotKey, set[str]] = {}
    for group_idx, blocks in enumerate(block_groups):
        for block_idx, layer in enumerate(blocks):
            region = _block_region(group_idx, block_idx)
            for slot in iter_buffer_slots(layer):
                regions.setdefault(slot.key, set()).add(region)
    return regions


def _param_storage_groups(
    model: nn.Module,
    slot_regions: dict[SlotKey, str],
) -> dict[tuple[object, ...], list[_ParamStorageMember]]:
    groups: dict[tuple[object, ...], list[_ParamStorageMember]] = {}
    for slot in iter_param_slots(model):
        param = slot.get()
        if param.numel() == 0:
            continue
        region = slot_regions.get(slot.key, _NON_BLOCK_REGION)
        groups.setdefault(param_storage_key(param), []).append(
            _ParamStorageMember(
                region=region,
                name=slot.name,
                requires_grad=param.requires_grad,
            )
        )
    return groups


def _buffer_storage_groups(
    model: nn.Module,
    slot_regions: dict[SlotKey, set[str]],
) -> dict[tuple[object, ...], list[_BufferStorageMember]]:
    groups: dict[tuple[object, ...], list[_BufferStorageMember]] = {}
    for slot in iter_buffer_slots(model):
        buffer = slot.get()
        if buffer.numel() == 0:
            continue
        regions = slot_regions.get(slot.key, {_NON_BLOCK_REGION})
        for region in regions:
            groups.setdefault(buffer_storage_key(buffer), []).append(
                _BufferStorageMember(region=region, name=slot.name)
            )
    return groups


def _validate_param_storage_group(
    members: list[_ParamStorageMember],
    *,
    stream_trainables: bool,
) -> None:
    regions = {member.region for member in members}
    names = sorted(member.name for member in members)
    grad_flags = {member.requires_grad for member in members}
    if len(grad_flags) > 1:
        raise ValueError(
            f"Tied storage spans both trainable and frozen parameters: "
            f"{names}. Slot-replace (frozen) and storage-swap "
            "(trainable) mechanisms cannot share a tied storage. "
            "Untie the parameters or freeze/unfreeze them consistently."
        )
    requires_grad = next(iter(grad_flags))
    if requires_grad:
        if stream_trainables and len(regions) > 1:
            raise ValueError(
                f"All-trainable tied storage spans streamed regions "
                f"{sorted(regions)}: {names}. Shared trainable storage "
                "across independently streamed regions creates divergent "
                "pinned clones; per-block .data swap and optimizer-step "
                "copy-back cannot coordinate to preserve the alias. Untie "
                "the parameters, refactor so the alias is intra-region, or use "
                "stream_trainable_weights=False."
            )
        return
    _validate_frozen_param_storage_group(
        regions=regions,
        names=names,
    )


def _validate_frozen_param_storage_group(
    *,
    regions: set[str],
    names: list[str],
) -> None:
    if len(regions) > 1:
        raise ValueError(
            f"Block streaming does not support tied parameters across "
            f"streamed regions: storage shared by {names}. Slot-local "
            "block streaming cannot preserve cross-region tying. Use "
            "whole-model PinnedWeights, disable block streaming, or "
            "untie the parameters."
        )


def _validate_buffer_storage_group(
    members: list[_BufferStorageMember],
) -> None:
    regions = {member.region for member in members}
    names = sorted({member.name for member in members})
    if len(regions) > 1:
        raise ValueError(
            f"Block streaming does not support tied buffers across "
            f"streamed regions: storage shared by {names}. The two "
            "pinning regimes (per-block clone vs composed "
            "PinnedWeights) can't coordinate to preserve the alias. "
            "Untie the buffers or use whole-model PinnedWeights "
            "instead."
        )
