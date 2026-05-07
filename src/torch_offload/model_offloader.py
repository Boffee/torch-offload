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
from collections.abc import Sequence
from types import TracebackType
from typing import Literal

import torch
from torch import nn

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
    return module.weight.dtype


class ModelOffloader:
    """Stream transformer blocks between pinned CPU and GPU with
    optional LoRA merge and trainable-parameter support.

    Composes :class:`PinnedWeights` (non-block frozen params),
    :class:`TrainableWeights` (LoRA / adapter params), and one or more
    :class:`StreamedWeights`\\ s internally. LoRA transforms are set via
    :meth:`set_loras` and attach to individual
    :class:`PinnedParamBuffer` objects so the merge fires automatically
    during DMA — no separate merge strategy needed.

    Parameters
    ----------
    model:
        The model containing the block list(s). Must be on CPU.
    target_device:
        GPU device for inference / training.
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
    strict_homogeneous:
        Forwarded to each :class:`StreamedWeights`. When True (default),
        non-homogeneous groups raise at construction. Pass False for
        the per-load-allocation fallback.
    """

    def __init__(
        self,
        model: nn.Module,
        target_device: torch.device,
        *,
        layers_attr: str | Sequence[str],
        blocks_to_swap: int | Sequence[int],
        prefetch_count: int | Sequence[int] = 2,
        cyclic: bool = False,
        strict_homogeneous: bool = True,
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

        detect_streaming_region_ties(model, block_groups)
        model.to("cpu")

        trainable_slots: set[SlotOwnership] = {
            s.slot for s in iter_param_slots(model) if s.param.requires_grad
        }

        streamers: list[StreamedWeights] = []
        for i, blocks in enumerate(block_groups):
            streamers.append(
                StreamedWeights(
                    blocks=blocks,
                    target_device=target_device,
                    blocks_to_swap=swap_list[i],
                    prefetch_count=pf_list[i],
                    cyclic=cyclic,
                    name=f"StreamedWeights[{layer_paths[i]}]",
                    strict_homogeneous=strict_homogeneous,
                    skip_slots=trainable_slots,
                )
            )

        skip_slots: set[SlotOwnership] = set(trainable_slots)
        for s in streamers:
            skip_slots |= s.slot_filter

        non_block: PinnedWeights | None = None
        has_pinnable = any(
            s.slot not in skip_slots for s in iter_param_slots(model)
        ) or any(
            s.slot not in skip_slots for s in iter_buffer_slots(model)
        )
        if has_pinnable:
            non_block = PinnedWeights(model, target_device, skip_slots=skip_slots)

        components: list[ModelStrategyComponent] = []
        if non_block is not None:
            components.append(non_block)
        components.append(TrainableWeights(model, target_device))
        components.extend(streamers)

        self._model = model
        self._target_device = target_device
        self._components = components
        self._streamers = streamers
        self._teardown_stack: contextlib.ExitStack | None = None

        self._reverse_index, self._reverse_parents = self._build_reverse_index(
            streamers, layer_paths, non_block,
        )
        # Pending routed-mode LoRAs: list of (parent_module, refs).
        # Populated by set_loras(mode="routed"); the actual hooks are
        # installed on activate() and removed on deactivate(). Cleared
        # on every set_loras() call (including merge mode and clears).
        self._pending_routes: list[
            tuple[nn.Module, list[tuple[torch.Tensor, torch.Tensor, float]]]
        ] = []

    # ------------------------------------------------------------------ API

    def set_loras(
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
          parent module. Forward becomes ``y = base(x) + α·B·A·x``.
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
                if mode == "merge" and buf.cpu_param.dtype not in _ADDMM_DTYPES:
                    raise ValueError(
                        f"LoRA target {target_key!r} has dtype "
                        f"{buf.cpu_param.dtype}; addmm_ merge requires "
                        f"bf16, fp16, or fp32. Use mode='routed' (or "
                        f"PEFT routed mode) for quantized params."
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
                if len(parents) > 1:
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
                parent = parents[0]
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

    def activate(self) -> None:
        try:
            with contextlib.ExitStack() as stack:
                for component in self._components:
                    stack.callback(component.deactivate)
                    component.activate()
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
                        parent, refs, self._target_device,
                        dtype=_routed_factor_dtype(parent),
                    )
                    stack.callback(handle.remove)
                self._teardown_stack = stack.pop_all()
        except BaseException:
            for buf in self._reverse_index.values():
                buf.transform = None
            self._pending_routes = []
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

    def __enter__(self) -> nn.Module:
        self.activate()
        return self.model

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.deactivate()

    # ----------------------------------------------------------- Internals

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
            block_locs_per = streamer.param_locs_per_block
            for block_idx, (block_bufs, block_locs) in enumerate(
                zip(block_bufs_per, block_locs_per, strict=True)
            ):
                for buf, (qual_name, parent, _leaf) in zip(
                    block_bufs, block_locs, strict=True
                ):
                    full_name = f"{layer_path}.{block_idx}.{qual_name}"
                    key = canonical_param_name(full_name)
                    bufs[key] = buf
                    parents[key] = (parent,)

        if non_block is not None:
            for buf, locs in non_block.slots:
                key = canonical_param_name(buf.name)
                bufs[key] = buf
                if locs:
                    parents[key] = tuple(p for p, _leaf in locs)

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


# ---------------------------------------------------------------------------
# Cross-region tied-weight detection
# ---------------------------------------------------------------------------


def detect_streaming_region_ties(  # noqa: PLR0912
    model: nn.Module, block_groups: Sequence[Sequence[nn.Module]]
) -> None:
    """Raise if any frozen storage is shared across streaming regions.

    Each entry in ``block_groups`` is one block list — one "region"
    per block. Everything else in ``model`` is the "non_block"
    region.

    Three configurations are unsupported:

    - **Cross-region ties** (block<->block, block<->non-block): the per-
      region pinning regimes can't coordinate to share storage.
    - **Mixed frozen/trainable ties** anywhere — across regions OR
      within a single region. The frozen side gets pinned and slot-
      replaced while the trainable side is moved via storage swap on
      activate; the two mechanisms cannot share a tied storage
      without breaking aliasing invariants silently.
    - **Intra-block ties** (two slots in the same block sharing
      storage): per-block stores walk ``named_parameters()`` with
      default duplicate removal and only swap one alias slot,
      leaving the other pointing at non-pinned data.

    Non-block-internal all-frozen ties (the standard ``tie_weights()``
    embed<->head pattern) are handled correctly by
    :class:`PinnedWeights`'s storage-key dedup and are NOT rejected
    here.
    """
    param_id_to_region: dict[int, str] = {}
    for group_idx, blocks in enumerate(block_groups):
        for block_idx, layer in enumerate(blocks):
            for p in layer.parameters():
                param_id_to_region.setdefault(
                    id(p), f"block:{group_idx}:{block_idx}"
                )

    groups: dict[tuple, list[tuple[str, str, bool, int, str, int]]] = {}
    for s in iter_param_slots(model):
        if s.param.numel() == 0:
            continue
        region = param_id_to_region.get(id(s.param), "non_block")
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
