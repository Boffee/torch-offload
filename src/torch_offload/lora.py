"""LoRA types and per-weight merge / routed transforms.

:class:`LoRA` pairs, validates, and pins factor matrices from a flat
safetensors state dict at construction.  The raw state dict is not
retained — this object owns the only copy of the pinned factors.

Two application paths share the same :class:`LoRA` data container:

- :class:`LoRATransform` (merge mode) — applied via in-place ``addmm_``
  on the GPU weight buffer after DMA; integrates with block streaming.
  Requires the base weight to be float (bf16/fp16/fp32).
- :class:`LoRARouteHandle` (routed mode) — installs a forward hook on
  the layer that adds ``α · (x @ A.T @ B.T)`` to the layer's output;
  base weight is not touched in place. Restricted to ``nn.Linear``
  parents (other layer types raise) and tied weights are rejected.
  Compatible with quantized bases whose ``weight.dtype`` reports the
  compute dtype (quanto ``WeightQBytesTensor``) or that expose
  ``module.compute_dtype`` (BitsAndBytes ``Linear4bit``); formats
  that report storage int via ``weight.dtype`` and provide no
  module-level compute dtype (``Linear8bitLt``, GGUF) need a forward
  probe and aren't covered here. For richer per-format LoRA coverage,
  prefer PEFT's per-type LoraLayer subclasses.

:class:`~torch_offload.ModelOffloader` is the consumer-facing API; its
``set_loras(..., mode=...)`` picks the path. The merge path fires
during DMA via :class:`LoRATransform`; the routed path lives as
forward hooks installed on activate and removed on deactivate.
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType

import torch
from torch import nn

__all__ = [
    "KeyTransformT",
    "LoRA",
    "LoRARouteHandle",
    "LoRATransform",
    "default_key_transform",
]


KeyTransformT = Callable[[str], str] | None


def default_key_transform(key: str) -> str:
    """Strip the common ``diffusion_model.`` prefix from ComfyUI LoRA keys."""
    prefix = "diffusion_model."
    return key[len(prefix) :] if key.startswith(prefix) else key


class LoRA:
    """A LoRA adapter with pinned factor matrices.

    Factors are paired, validated, and pinned to host memory at
    construction.  The raw ``state_dict`` is not retained.

    Implements :class:`~torch_offload.protocols.CachedResource` so it
    can be registered in :class:`~torch_offload.ModelCache` for budget
    tracking and LRU eviction.  ``activate``/``deactivate`` are no-ops
    — factors stay on pinned CPU and are copied to GPU per-parameter
    by :class:`LoRATransform` during the merge-on-DMA callback.

    Strength is extrinsic — specify it when passing the adapter to
    :meth:`ModelOffloader.set_loras` as a ``(LoRA, strength)`` tuple.

    ``key_transform`` is applied to state-dict keys before pairing.
    Defaults to stripping the ``diffusion_model.`` prefix common in
    ComfyUI LoRA files.  Pass ``None`` to disable.
    """

    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        key_transform: KeyTransformT = default_key_transform,
    ) -> None:
        self._factors = _pair_and_pin(state_dict, key_transform)

    @property
    def targets(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Map of target weight name to (A_pinned, B_pinned)."""
        return self._factors

    @property
    def cache_bytes(self) -> int:
        return sum(a.nbytes + b.nbytes for a, b in self._factors.values())

    @property
    def value(self) -> LoRA:
        return self

    def activate(self) -> None:
        pass

    def deactivate(self) -> None:
        pass

    def __enter__(self) -> LoRA:
        self.activate()
        return self.value

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.deactivate()


class LoRATransform:
    """Per-weight LoRA factors applied after DMA to GPU.

    Holds references to LoRA-owned pinned factor matrices — no cloning
    or pinning happens here.  :meth:`apply` copies each factor pair to
    GPU and merges via ``addmm_`` with ``alpha=strength``.
    """

    __slots__ = ("_refs",)

    def __init__(
        self, refs: list[tuple[torch.Tensor, torch.Tensor, float]]
    ) -> None:
        self._refs = refs

    def apply(self, gpu_data: torch.Tensor) -> None:
        dev, dt = gpu_data.device, gpu_data.dtype
        for a, b, strength in self._refs:
            a_gpu = a.to(device=dev, dtype=dt, non_blocking=True)
            b_gpu = b.to(device=dev, dtype=dt, non_blocking=True)
            gpu_data.addmm_(b_gpu, a_gpu, alpha=strength)


class LoRARouteHandle:
    """Live forward-hook for one routed LoRA target.

    Owns GPU copies of the LoRA factors plus the registered hook on
    the parent module. Forward path becomes::

        y = base(x) + sum_i strength_i * (x @ A_i.T @ B_i.T)

    When the target has multiple LoRAs (merging multiple adapters
    against the same weight), factors are pre-fused at install time:

    - ``A_fused``  shape ``(sum_i r_i, in_dim)``     — A_i stacked along the rank axis
    - ``B_fused``  shape ``(out_dim, sum_i r_i)``    — B_i scaled by strength_i, stacked along the rank axis

    The math identity ``B_fused @ A_fused == sum_i strength_i * B_i @ A_i``
    falls out from the stacking: each ``m`` in the fused rank dimension
    lands in exactly one LoRA's slice. Mixed ranks are handled
    naturally — the fused rank is ``sum_i r_i`` rather than ``N · r``.

    Single fused GEMM at the combined rank replaces N per-LoRA GEMMs:
    fewer kernel launches, better tensor-core utilization, better
    memory locality.

    The fused tensors are preallocated empty and filled by per-slice
    pinned-CPU → GPU DMA (``copy_`` casts dtype during the transfer);
    strength is baked via in-place ``mul_`` on the slice. This avoids
    the staging-then-cat path's transient peak (one per-LoRA GPU
    tensor *plus* the fused result both live momentarily).

    Construction installs the hook; :meth:`remove` removes it and
    drops the GPU factor refs so refcount-GC reclaims them.

    Restricted to ``nn.Linear``-shaped forwards. The math assumes
    ``base(x) = x @ W.T (+ bias)``; LoRA applied to Conv2d, Embedding,
    or other layouts needs different formulas (see PEFT's per-type
    LoraLayer subclasses).
    """

    __slots__ = ("_handle", "_a_fused", "_b_fused")

    def __init__(
        self,
        parent: nn.Module,
        refs: list[tuple[torch.Tensor, torch.Tensor, float]],
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> None:
        if not refs:
            raise ValueError("LoRARouteHandle requires at least one (A, B, strength) ref")

        # Pull shape from the first ref, then verify the rest agree.
        # All LoRAs against the same target share in_dim / out_dim by
        # construction (they parameterize the same base weight); we
        # validate defensively to fail loud on caller misuse.
        in_dim = refs[0][0].shape[1]
        out_dim = refs[0][1].shape[0]
        ranks = [a.shape[0] for a, _, _ in refs]
        for a, b, _ in refs:
            if a.shape[1] != in_dim or b.shape[0] != out_dim:
                raise ValueError(
                    f"LoRA factor shape mismatch: expected A=(r, {in_dim}), "
                    f"B=({out_dim}, r); got A={tuple(a.shape)}, "
                    f"B={tuple(b.shape)}."
                )
        total_rank = sum(ranks)
        factor_dtype = dtype if dtype is not None else refs[0][0].dtype

        # Preallocate the fused tensors, then DMA each per-LoRA factor
        # straight into its slice. `copy_` casts dtype as part of the
        # transfer, so factors stored as fp32 land as the layer's
        # bf16/fp16 in one operation. `mul_` bakes the strength
        # in-place — no intermediate scaled-B tensor.
        a_fused = torch.empty(
            (total_rank, in_dim), device=device, dtype=factor_dtype,
        )
        b_fused = torch.empty(
            (out_dim, total_rank), device=device, dtype=factor_dtype,
        )
        offset = 0
        for (a, b, strength), r in zip(refs, ranks, strict=True):
            a_fused[offset : offset + r].copy_(a, non_blocking=True)
            b_slice = b_fused[:, offset : offset + r]
            b_slice.copy_(b, non_blocking=True)
            b_slice.mul_(strength)
            offset += r

        self._a_fused = a_fused
        self._b_fused = b_fused

        def hook(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            x = inputs[0]
            # x: (..., in_dim); A_fused: (R, in_dim); B_fused: (out_dim, R)
            # where R = sum_i rank_i. Contribution: x @ A.T @ B.T.
            return output + (x @ a_fused.T) @ b_fused.T

        self._handle = parent.register_forward_hook(hook)

    def remove(self) -> None:
        self._handle.remove()
        # Drop the GPU factor refs. The closure also holds them via
        # the captured locals, but unregistering removes the hook
        # function from the module's _forward_hooks dict, so the
        # closure becomes unreachable and Python refcount-GCs it.
        self._a_fused = None
        self._b_fused = None


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _pair_and_pin(
    state_dict: dict[str, torch.Tensor],
    key_transform: KeyTransformT,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    a_tensors: dict[str, torch.Tensor] = {}
    b_tensors: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.endswith(".lora_A.weight"):
            base_key = key[: -len(".lora_A.weight")]
            a_tensors[base_key] = tensor
        elif key.endswith(".lora_B.weight"):
            base_key = key[: -len(".lora_B.weight")]
            b_tensors[base_key] = tensor

    a_only = set(a_tensors) - set(b_tensors)
    b_only = set(b_tensors) - set(a_tensors)
    if a_only or b_only:
        raise ValueError(
            f"Unpaired LoRA factors: A-only={sorted(a_only)}, "
            f"B-only={sorted(b_only)}. Each target needs both "
            f".lora_A.weight and .lora_B.weight."
        )

    factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for base_key, a in a_tensors.items():
        b = b_tensors[base_key]
        target_key = f"{base_key}.weight"
        if key_transform is not None:
            target_key = key_transform(target_key)

        if not a.is_floating_point() or not b.is_floating_point():
            raise ValueError(
                f"LoRA factors for {target_key!r}: must be "
                f"floating-point; got A.dtype={a.dtype}, "
                f"B.dtype={b.dtype}."
            )
        if a.dim() != 2 or b.dim() != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(
                f"LoRA factor shape mismatch for {target_key!r}: "
                f"A.shape={tuple(a.shape)}, B.shape={tuple(b.shape)}. "
                f"Expected A=(rank, in_dim), B=(out_dim, rank) with "
                f"A.shape[0] == B.shape[1]."
            )

        if target_key in factors:
            raise ValueError(
                f"Duplicate LoRA target {target_key!r}: key_transform "
                f"mapped multiple source keys to the same target."
            )
        factors[target_key] = (
            a.contiguous().pin_memory(),
            b.contiguous().pin_memory(),
        )

    return factors
