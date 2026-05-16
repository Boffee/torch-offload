"""Unified block-streaming strategy with optional LoRA merge.

Composes block streaming, non-block pinning, trainable parameter
movement, and optional per-weight LoRA transforms into a single
:class:`ModelOffloader` class.

Also provides :func:`detect_streaming_region_ties`
(construction-time validation), used internally by
:class:`ModelOffloader` and exported for direct use / testing.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Literal

import torch
from torch import nn

from ._devices import canonical_device
from .lora import _ADDMM_DTYPES, LoRA, LoRARouteHandle, LoRATransform
from .pinned_buffer import PinnedParamBuffer, storage_key
from .pinned_weights import PinnedWeights
from .protocols import ModelStrategyComponent, SlotOwnership
from .slots import canonical_param_name, iter_buffer_slots, iter_param_slots, walk_attr_path
from .streamed_weights import StreamedWeights
from .trainable_weights import TrainableWeights

logger = logging.getLogger(__name__)

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
    - ``module.weight.dtype`` — plain fp16/bf16/fp32, plus quanto
      (``WeightQBytesTensor.dtype`` is the scale dtype, not the int8
      storage of ``_data``).

    Formats whose ``weight.dtype`` reports the storage int (e.g.,
    BitsAndBytes ``Linear8bitLt``'s ``Int8Params``, GGUF's packed
    ``uint8``) and which don't expose ``compute_dtype`` would need a
    forward probe; not handled here.
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
    return weight.dtype


class ModelOffloader:
    """Stream transformer blocks between pinned CPU and CUDA with
    optional LoRA merge and trainable-parameter support.

    CUDA activation uses block streaming plus component-level device
    movement. CPU activation is pass-through over the pinned
    host-backed module slots.

    Composes :class:`PinnedWeights` (non-block frozen params),
    :class:`TrainableWeights` (LoRA / adapter params), and one or more
    :class:`StreamedWeights`\\ s internally. LoRA transforms are set via
    :meth:`set_loras` and attach to individual
    :class:`PinnedParamBuffer` objects so the merge fires automatically
    during DMA — no separate merge strategy needed.

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

    By default, trainable params keep the historical CUDA behavior:
    all trainables, including adapters inside streamed blocks, stay
    GPU-resident while the offloader is active on CUDA and work with
    normal PyTorch optimizers. CPU activation leaves them in the
    host-backed module slots.

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
        The model containing the block list(s). Must be on CPU.
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
        Default ``False`` preserves the historical contract:
        all trainable params are skipped by block streaming and managed
        by :class:`TrainableWeights`, so normal ``optimizer.step()``
        works unchanged. ``True`` streams in-block trainable parameter
        data with the block residency manager; use :meth:`optimizer_step`
        around the optimizer update.
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

        _check_block_groups_disjoint(block_groups, layer_paths)
        detect_streaming_region_ties(
            model, block_groups, stream_trainables=stream_trainable_weights,
        )
        model.to("cpu")

        trainable_slots: set[SlotOwnership] = {
            s.slot for s in iter_param_slots(model) if s.param.requires_grad
        }

        # By default, streamers skip trainables and TrainableWeights keeps
        # them all GPU-resident. With stream_trainable_weights=True,
        # streamers handle in-block params of both kinds: frozen via slot
        # replacement and trainable via identity-preserving ``.data`` swap.
        # Gradients are NOT streamed — they live on GPU during backward via
        # PyTorch's native ``AccumulateGrad`` mechanism.
        streamer_skip_slots = None if stream_trainable_weights else trainable_slots
        streamers: list[StreamedWeights] = []
        for i, blocks in enumerate(block_groups):
            streamers.append(
                StreamedWeights(
                    blocks=blocks,
                    blocks_to_swap=swap_list[i],
                    prefetch_count=pf_list[i],
                    cyclic=cyclic,
                    name=f"StreamedWeights[{layer_paths[i]}]",
                    skip_slots=streamer_skip_slots,
                )
            )

        streamer_slots: set[SlotOwnership] = set()
        for s in streamers:
            streamer_slots |= s.slot_filter

        # PinnedWeights manages frozen non-block params/buffers. Skip
        # everything the streamers own (in-block, frozen + trainable)
        # plus any remaining out-of-block trainables (those go to
        # TrainableWeights). Equivalent to ``streamer_slots |
        # trainable_slots`` since streamer_slots already covers in-
        # block trainables.
        pinned_skip_slots = streamer_slots | trainable_slots

        non_block: PinnedWeights | None = None
        has_pinnable = any(
            s.slot not in pinned_skip_slots for s in iter_param_slots(model)
        ) or any(
            s.slot not in pinned_skip_slots for s in iter_buffer_slots(model)
        )
        if has_pinnable:
            non_block = PinnedWeights(
                model, skip_slots=pinned_skip_slots,
            )

        # TrainableWeights handles only out-of-block trainables (in-
        # block trainables stream through the slot pool with .data
        # swap). Skip the streamer's territory; the requires_grad
        # filter inside TrainableWeights handles frozen slots in
        # streamer_slots automatically.
        components: list[ModelStrategyComponent] = []
        if non_block is not None:
            components.append(non_block)
        if stream_trainable_weights:
            components.append(
                TrainableWeights(model, skip_slots=streamer_slots),
            )
        else:
            components.append(TrainableWeights(model))
        components.extend(streamers)

        self._model = model
        self._active_device: torch.device | None = None
        self._components = components
        self._streamers = streamers
        self._teardown_stack: contextlib.ExitStack | None = None

        self._reverse_index, self._reverse_parents = self._build_reverse_index(
            streamers, layer_paths, non_block,
        )
        self._block_groups: list[list[nn.Module]] = block_groups
        self._warned_about_checkpointing: bool = False
        self._stream_trainable_weights: bool = stream_trainable_weights
        self._skip_checkpointing_check: bool = skip_checkpointing_check
        self._is_block_checkpointed: Callable[[nn.Module], bool] = (
            is_block_checkpointed
            if is_block_checkpointed is not None
            else _hf_block_has_checkpointing_flag
        )
        # Pending routed-mode LoRAs: list of (parent_module, refs).
        # Populated by set_loras(mode="routed"); the actual hooks are
        # installed on activate() and removed on deactivate(). Cleared
        # on every set_loras() call (including merge mode and clears).
        self._pending_routes: list[
            tuple[nn.Module, list[tuple[torch.Tensor, torch.Tensor, float]]]
        ] = []

    # ------------------------------------------------------------------ API

    def set_loras(  # noqa: PLR0912
        self,
        loras: Sequence[tuple[LoRA, float]],
        *,
        mode: Literal["merge", "routed"] = "merge",
    ) -> None:
        """Attach LoRAs for the next activation cycle.

        Must be called while deactivated. Attachments are cleared
        automatically on :meth:`deactivate`, so callers must call
        ``set_loras`` before each :meth:`activate` if they want
        LoRA-augmented inference.

        ``mode``:

        - ``"merge"`` (default): attaches a :class:`LoRATransform` per
          matched :class:`PinnedParamBuffer`. The merge fires during
          DMA via in-place ``addmm_``, so it rides along with the
          streaming cycle. Requires base weight dtype to be bf16, fp16,
          or fp32.
        - ``"routed"``: registers a forward hook on each matched
          parent module. Forward becomes ``y = base(x) + alpha * B * A * x``.
          Doesn't touch the base weight in place. Restricted to
          ``nn.Linear`` parents (the math assumes ``y = x @ W.T``);
          tied targets and non-Linear parents raise. Quantized bases
          work when the compute dtype is probeable via
          :func:`_routed_factor_dtype` — covers plain fp/bf16/fp16,
          quanto (``weight.dtype`` already reports scale dtype), and
          formats that expose ``module.compute_dtype`` (BitsAndBytes
          ``Linear4bit``). Formats that report storage int via
          ``weight.dtype`` without a module-level compute dtype
          (``Linear8bitLt``, GGUF) aren't covered.

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
        # Clear any prior attachments from either mode.
        for buf in self._reverse_index.values():
            buf.transform = None
        self._pending_routes = []

        if not loras:
            return

        per_target: dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]] = {}
        per_buffer_target: dict[int, str] = {}
        total_targets = 0
        matched_targets = 0
        for lora, strength in loras:
            for target_key, (a, b) in lora.targets.items():
                total_targets += 1
                buf = self._reverse_index.get(target_key)
                if buf is None:
                    continue
                matched_targets += 1
                expected = tuple(buf.cpu_param.shape)
                if expected != (b.shape[0], a.shape[1]):
                    raise ValueError(
                        f"LoRA factor shape mismatch for {target_key!r}: "
                        f"B@A produces ({b.shape[0]}, {a.shape[1]}), "
                        f"target shape is {expected}."
                    )
                existing_target = per_buffer_target.setdefault(id(buf), target_key)
                if existing_target != target_key:
                    raise ValueError(
                        f"LoRA targets {existing_target!r} and {target_key!r} "
                        f"resolve to the same tied parameter storage. Apply "
                        f"only one alias for a tied weight in a single "
                        f"set_loras() call; otherwise the same base weight "
                        f"would receive multiple logical updates."
                    )
                if mode == "merge":
                    cpu_data = buf.cpu_param.data
                    # Two distinct rejection paths for merge mode:
                    # - Subclassed tensors (quanto WeightQBytesTensor,
                    #   GGUFWeight, etc.) advertise a float dtype but
                    #   silently drop in-place ops on the wrapper —
                    #   addmm_ on a quanto tensor returns success while
                    #   leaving the int8 _data unchanged.
                    # - Plain float tensors must be in the addmm-capable
                    #   dtype set.
                    if type(cpu_data) is not torch.Tensor:
                        raise ValueError(
                            f"LoRA target {target_key!r} is wrapped in "
                            f"{type(cpu_data).__name__}; merge mode "
                            f"requires a plain torch.Tensor (in-place "
                            f"addmm_ on subclassed tensors silently "
                            f"drops the update). Use mode='routed' (or "
                            f"PEFT routed mode), or apply "
                            f"torch_offload.merge_lora() permanently."
                        )
                    if cpu_data.dtype not in _ADDMM_DTYPES:
                        raise ValueError(
                            f"LoRA target {target_key!r} has dtype "
                            f"{cpu_data.dtype}; addmm_ merge requires "
                            f"bf16, fp16, or fp32. Use mode='routed' (or "
                            f"PEFT routed mode) for non-float bases."
                        )
                per_target.setdefault(target_key, []).append(
                    (a, b, strength)
                )

        if matched_targets < total_targets:
            sample_lora = sorted(next(iter(loras))[0].targets)[:3]
            sample_index = sorted(self._reverse_index)[:3]
            logger.warning(
                "set_loras matched %d/%d targets. "
                "Sample LoRA keys: %s ... Sample index keys: %s ...",
                matched_targets, total_targets, sample_lora, sample_index,
            )
        else:
            logger.debug("set_loras matched %d/%d targets", matched_targets, total_targets)

        if mode == "merge":
            for target_key, refs in per_target.items():
                self._reverse_index[target_key].transform = LoRATransform(refs)
        else:  # routed
            # Two-pass: validate every parent before mutating state, so
            # a mid-list rejection (non-Linear, tied weight) leaves the
            # offloader in the cleared state set above instead of with
            # a half-built _pending_routes the caller can't see.
            new_routes: list[
                tuple[nn.Module, list[tuple[torch.Tensor, torch.Tensor, float]]]
            ] = []
            for target_key, refs in per_target.items():
                parents = self._reverse_parents[target_key]
                if len(parents) != 1:
                    raise ValueError(
                        f"Routed LoRA mode does not support tied "
                        f"weights; target {target_key!r} has "
                        f"{len(parents)} parent locations (typically the "
                        f"tied embed/head pattern). The hook would only "
                        f"fire on one of them, silently missing the "
                        f"others. Use mode='merge' for tied targets — "
                        f"merge mutates the shared storage so all "
                        f"locations see the LoRA contribution."
                    )
                parent = next(iter(parents))
                if not isinstance(parent, nn.Linear):
                    raise ValueError(
                        f"Routed LoRA mode requires nn.Linear targets; "
                        f"target {target_key!r} has parent module of "
                        f"type {type(parent).__name__}. Use mode='merge' "
                        f"for non-Linear targets, or wrap the model with "
                        f"PEFT for richer per-type routing."
                    )
                new_routes.append((parent, refs))
            self._pending_routes = new_routes

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
        active_device = self._resolve_device(device)
        if active_device.type == "cpu":
            self._validate_no_merge_transforms_for_cpu()
        elif active_device.type == "cuda":
            self._enforce_checkpointing_for_trainable_streaming()
            self._warn_if_training_without_checkpointing()
        else:
            raise ValueError(
                "ModelOffloader.activate() supports CUDA or CPU; "
                f"got {active_device}."
            )
        self._active_device = active_device
        try:
            with contextlib.ExitStack() as stack:
                for component in self._components:
                    stack.callback(component.deactivate)
                    component.activate(active_device)
                # Install routed-mode hooks AFTER components activate
                # so the LIFO ExitStack unwinds them FIRST on
                # deactivate — routes come down before component
                # teardown, which means the hook is gone by the time
                # streamers reset slot Parameters. Cast factors to the
                # parent's compute dtype (see _routed_factor_dtype) —
                # LoRA state-dicts are typically fp32 and the base may
                # be bf16, fp16, or quantized; the hook math needs
                # factors in the same dtype the layer outputs.
                for parent, refs in self._pending_routes:
                    handle = LoRARouteHandle(
                        parent, refs, active_device,
                        dtype=_routed_factor_dtype(parent),
                    )
                    stack.callback(handle.remove)
                self._teardown_stack = stack.pop_all()
        except BaseException:
            for buf in self._reverse_index.values():
                buf.transform = None
            self._pending_routes = []
            self._active_device = None
            raise

    def deactivate(self) -> None:
        stack = self._teardown_stack
        self._teardown_stack = None
        try:
            if stack is not None:
                stack.close()
        finally:
            for buf in self._reverse_index.values():
                buf.transform = None
            self._pending_routes = []
            self._active_device = None

    @contextlib.contextmanager
    def optimizer_step(self) -> Iterator[None]:
        """Context manager wrapping the optimizer-step boundary for
        streamed trainable weights.

        On CUDA activation, brings every streamer-managed trainable's
        ``.data`` to GPU on enter (force-evicting any currently-loaded
        blocks first to normalize state), yields, and on exit D2H's the
        post-step ``.data`` back to the pinned host clones (blocking, to
        avoid racing the next iteration's prefetch). On CPU activation,
        this is a guarded no-op.

        ``param.grad`` is unaffected throughout. On CUDA, it lives on
        GPU during backward via PyTorch's native ``AccumulateGrad`` and
        is read+modified by the optimizer in place. ``optimizer.zero_grad()``,
        ``clip_grad_norm_``, AMP's ``GradScaler.unscale_`` and other
        grad-walking tools work as in vanilla PyTorch — they don't need
        to be inside this context.

        Out-of-block trainables (handled by ``TrainableWeights``) are
        also unaffected; on CUDA they're GPU-resident across the whole
        activation cycle, just like the default
        ``stream_trainable_weights=False`` path.

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
            for streamer in self._streamers:
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

    def _validate_no_merge_transforms_for_cpu(self) -> None:
        if any(buf.transform is not None for buf in self._reverse_index.values()):
            raise ValueError(
                "ModelOffloader merge transforms require CUDA activation; "
                "got cpu. Use set_loras(..., mode='routed') for "
                "CPU activation, or activate on CUDA."
            )

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
    def _build_reverse_index(
        streamers: list[StreamedWeights],
        layer_paths: list[str],
        non_block: PinnedWeights | None,
    ) -> tuple[dict[str, PinnedParamBuffer], dict[str, tuple[nn.Module, ...]]]:
        """Map canonical param names to their :class:`PinnedParamBuffer`
        and to the tuple of parent modules where the param is installed.

        Two parallel dicts: the buffer map drives merge-mode LoRA
        (transform attached to the buffer), and the parents map drives
        routed-mode LoRA (forward hook on the parent layer).

        Keys are normalized to strip PEFT's ``.base_layer.`` segments
        so that LoRA state-dict keys (which use the original model
        names) match regardless of whether the model is PEFT-wrapped.

        Parents are stored as a tuple to preserve tied-weight
        information. Streamed-block targets always have exactly one
        parent (each ``(layer_path, block_idx, qual_name)`` is unique;
        cross-region ties are rejected upstream by
        :func:`detect_streaming_region_ties`). Non-block targets may
        have more than one parent — e.g., the standard tied
        embed/head pattern — which routed mode rejects (merge mode
        handles tied storage uniformly because it mutates the shared
        bytes).
        """
        bufs: dict[str, PinnedParamBuffer] = {}
        parents: dict[str, tuple[nn.Module, ...]] = {}

        for streamer, layer_path in zip(streamers, layer_paths, strict=True):
            block_bufs_per = streamer.param_bufs_per_block
            block_aliases_per = streamer.param_aliases_per_block
            for block_idx, (block_bufs, block_aliases) in enumerate(
                zip(block_bufs_per, block_aliases_per, strict=True)
            ):
                for buf, aliases in zip(block_bufs, block_aliases, strict=True):
                    seen_parent_ids: set[int] = set()
                    alias_parents: list[nn.Module] = []
                    for _qual_name, parent, _leaf in aliases:
                        parent_id = id(parent)
                        if parent_id in seen_parent_ids:
                            continue
                        seen_parent_ids.add(parent_id)
                        alias_parents.append(parent)
                    parent_tuple = tuple(alias_parents)
                    for qual_name, _parent, _leaf in aliases:
                        full_name = f"{layer_path}.{block_idx}.{qual_name}"
                        key = canonical_param_name(full_name)
                        bufs[key] = buf
                        parents[key] = parent_tuple

        if non_block is not None:
            for buf, aliases in non_block.param_aliases:
                if not aliases:
                    continue
                seen_parent_ids: set[int] = set()
                alias_parents: list[nn.Module] = []
                for _name, parent, _leaf in aliases:
                    parent_id = id(parent)
                    if parent_id in seen_parent_ids:
                        continue
                    seen_parent_ids.add(parent_id)
                    alias_parents.append(parent)
                parent_tuple = tuple(alias_parents)
                for name, _parent, _leaf in aliases:
                    key = canonical_param_name(name)
                    bufs[key] = buf
                    parents[key] = parent_tuple

        return bufs, parents


# ---------------------------------------------------------------------------
# Module-private helpers (used only by ModelOffloader constructor)
# ---------------------------------------------------------------------------

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


def _check_block_groups_disjoint(
    block_groups: Sequence[Sequence[nn.Module]], layer_paths: Sequence[str]
) -> None:
    """Reject configurations where any :class:`SlotOwnership` is owned
    by more than one streamer region (or appears twice in a single
    region).

    Each ``StreamedWeights`` instance assumes it is the sole owner of
    every ``(parent_module, leaf, kind)`` slot inside its blocks.
    Overlapping ownership comes in three shapes:

    - **Same block module listed in two groups** (or twice in one):
      both streamers would pin and slot-replace the same ``nn.Module``,
      and per-block ``.data`` swap during streaming would race.
    - **Parent/child blocks across groups** (group A has a parent
      module, group B has one of its children): different
      ``id(parent)`` block instances, but their per-leaf
      ``SlotOwnership`` tuples coincide on the child's slots.
    - **Within a single group, the same nn.Module appearing twice**:
      the streamer would create per-index pinned clones for the same
      slots and swap ``.data`` redundantly.

    This check is the precondition for
    :func:`detect_streaming_region_ties` and the per-block pinning
    walk inside :class:`_BlockPinnedStore` to be unambiguous; it lives
    at the composer because it is about how the caller composed
    ``layers_attr`` paths, not about parameter aliasing.
    """
    slot_owner: dict[SlotOwnership, str] = {}
    duplicates: dict[SlotOwnership, list[str]] = {}
    for group_idx, blocks in enumerate(block_groups):
        for block_idx, layer in enumerate(blocks):
            label = f"{layer_paths[group_idx]}[{block_idx}]"
            slots: set[SlotOwnership] = set()
            for s in iter_param_slots(layer):
                slots.add(s.slot)
            for s in iter_buffer_slots(layer):
                slots.add(s.slot)
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
            f"block locations. Example: leaf {sample[0].leaf!r} is "
            f"owned by {sample[1]}. Likely causes: the same block is "
            f"listed in two `layers_attr` paths, a path resolves to a "
            f"module that contains another resolved path, or a block "
            f"appears twice in a single `nn.ModuleList`."
        )


# ---------------------------------------------------------------------------
# Cross-region tied-weight detection
# ---------------------------------------------------------------------------


def detect_streaming_region_ties(  # noqa: PLR0912
    model: nn.Module,
    block_groups: Sequence[Sequence[nn.Module]],
    *,
    stream_trainables: bool = False,
) -> None:
    """Raise if any frozen storage is shared across streaming regions.

    Each entry in ``block_groups`` is one block list — one "region"
    per block. Everything else in ``model`` is the "non_block"
    region.

    Unsupported configurations:

    - **Cross-region ties** (block<->block, block<->non-block): the per-
      region pinning regimes can't coordinate to share storage.
    - **Mixed frozen/trainable ties** anywhere — across regions OR
      within a single region. The frozen side gets pinned and slot-
      replaced while the trainable side is moved via storage swap on
      activate; the two mechanisms cannot share a tied storage
      without breaking aliasing invariants silently.
    - **Streamed trainable ties across regions** when
      ``stream_trainables=True``. The same Parameter object is safe
      under default all-trainables-on-GPU movement, but unsafe when
      one streamed region owns a distinct pinned clone.
    - **Unsupported intra-block ties** (two distinct slots in the
      same block sharing storage). Same-Parameter all-trainable
      aliases are safe because a single ``.data`` swap reaches every
      alias slot; frozen aliases and distinct-Parameter trainable
      aliases are rejected.

    Non-block-internal all-frozen ties (the standard ``tie_weights()``
    embed<->head pattern) are handled correctly by
    :class:`PinnedWeights`'s storage-key dedup and are NOT rejected
    here.
    """
    # Slot-level region map. Distinguishes "this slot lives in block X"
    # from "this slot's Parameter is also referenced from block Y" — the
    # latter is what cross-region same-Parameter aliasing produces, and a
    # Parameter-id-only map (with setdefault) would silently collapse
    # both slots to the first block's region.
    slot_to_region: dict[tuple[int, str], str] = {}
    for group_idx, blocks in enumerate(block_groups):
        for block_idx, layer in enumerate(blocks):
            for s in iter_param_slots(layer):
                slot_to_region[(id(s.parent), s.leaf)] = (
                    f"block:{group_idx}:{block_idx}"
                )

    groups: dict[tuple, list[tuple[str, str, bool, int, str, int]]] = {}
    for s in iter_param_slots(model):
        if s.param.numel() == 0:
            continue
        region = slot_to_region.get((id(s.parent), s.leaf), "non_block")
        skey = storage_key(s.param.data)
        groups.setdefault(skey, []).append(
            (region, s.name, s.param.requires_grad, id(s.parent), s.leaf, id(s.param))
        )

    for members in groups.values():
        regions = {region for region, _, _, _, _, _ in members}
        names = sorted(name for _, name, _, _, _, _ in members)
        grads = {grad for _, _, grad, _, _, _ in members}
        if len(grads) > 1:
            raise ValueError(
                f"Tied storage spans both trainable and frozen parameters: "
                f"{names}. Slot-replace (frozen) and storage-swap "
                "(trainable) mechanisms cannot share a tied storage. "
                "Untie the parameters or freeze/unfreeze them consistently."
            )
        if all(grads):
            param_ids = {pid for _, _, _, _, _, pid in members}
            if len(param_ids) > 1:
                raise ValueError(
                    f"All-trainable tied storage with distinct Parameter "
                    f"objects: {names}. TrainableWeights moves each Parameter "
                    "independently via p.data = ... and would break the "
                    "storage alias on GPU. Untie the parameters or use "
                    "tie_weights() to share a single Parameter object."
                )
            # Same Parameter object, all-trainable. In default mode this
            # is fine across regions because TrainableWeights owns the
            # single Parameter object directly. With
            # stream_trainable_weights=True, cross-region aliasing is NOT
            # fine: each region builds its own ``_BlockPinnedStore`` (or
            # composes TrainableWeights for ``non_block``) with an
            # independent pinned clone.
            if stream_trainables and len(regions) > 1:
                raise ValueError(
                    f"All-trainable tied storage spans streamed regions "
                    f"{sorted(regions)}: {names}. The same trainable "
                    "Parameter aliased across regions creates divergent "
                    "per-region pinned clones; per-block .data swap and "
                    "optimizer-step materialize/scatter cannot coordinate to "
                    "preserve the alias. Untie the parameters, refactor "
                    "so the alias is intra-region, or use "
                    "stream_trainable_weights=False."
                )
            # Same-Parameter all-trainable ties that reach here are
            # either default-mode cross-region ties (handled by one
            # TrainableWeights mover) or intra-region streamed trainable ties
            # (handled by one _BlockPinnedStore clone).
            continue
        if len(regions) > 1:
            raise ValueError(
                f"Block streaming does not support tied parameters across "
                f"streamed regions: storage shared by {names}. Slot-local "
                "block streaming cannot preserve cross-region tying. Use "
                "whole-model PinnedWeights, disable block streaming, or "
                "untie the parameters."
            )
        sole_region = next(iter(regions))
        if sole_region.startswith("block:"):
            slot_locs = {(pid, leaf) for _, _, _, pid, leaf, _ in members}
            if len(slot_locs) > 1:
                raise ValueError(
                    f"Block streaming does not support intra-block tied "
                    f"parameters: storage shared by {names} within "
                    f"{sole_region}. _BlockPinnedStore cannot preserve "
                    "the tying invariant — one alias would stay pointing "
                    "at non-pinned data. Untie the parameters or use "
                    "whole-model PinnedWeights instead."
                )

    block_buffer_slot_regions: dict[tuple[int, str], set[str]] = {}
    for group_idx, blocks in enumerate(block_groups):
        for block_idx, layer in enumerate(blocks):
            for s in iter_buffer_slots(layer):
                block_buffer_slot_regions.setdefault(
                    (id(s.parent), s.leaf), set()
                ).add(f"block:{group_idx}:{block_idx}")

    buf_groups: dict[tuple, list[tuple[str, str, int]]] = {}
    for s in iter_buffer_slots(model):
        if s.buffer.numel() == 0:
            continue
        regions = block_buffer_slot_regions.get((id(s.parent), s.leaf), {"non_block"})
        for region in regions:
            buf_groups.setdefault(storage_key(s.buffer), []).append(
                (region, s.name, id(s.buffer))
            )

    for members in buf_groups.values():
        regions = {region for region, _, _ in members}
        names = sorted({name for _, name, _ in members})
        if len(regions) > 1:
            raise ValueError(
                f"Block streaming does not support tied buffers across "
                f"streamed regions: storage shared by {names}. The two "
                "pinning regimes (per-block clone vs composed "
                "PinnedWeights) can't coordinate to preserve the alias. "
                "Untie the buffers or use whole-model PinnedWeights "
                "instead."
            )
        sole_region = next(iter(regions))
        if sole_region.startswith("block:"):
            distinct_ids = {bid for _, _, bid in members}
            if len(distinct_ids) > 1:
                raise ValueError(
                    f"Block streaming does not support intra-block tied "
                    f"buffers: storage shared by {names} within "
                    f"{sole_region}. _BlockPinnedStore clones each "
                    "buffer independently — the alias would break. "
                    "Untie the buffers or use whole-model PinnedWeights."
                )
