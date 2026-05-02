"""LoRA types and per-weight merge transform.

:class:`LoRA` pairs, validates, and pins factor matrices from a flat
safetensors state dict at construction.  The raw state dict is not
retained — this object owns the only copy of the pinned factors.

:class:`LoRATransform` holds lightweight references to LoRA-owned
pinned factors and applies the merge via in-place ``addmm_`` after DMA.

:class:`~torch_offload.ModelOffloader` is the consumer-facing API:
its ``set_loras`` method matches LoRA targets to model parameters and
attaches a :class:`LoRATransform` per matched weight.  The transform
fires automatically when the buffer copies to GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType

import torch

__all__ = [
    "KeyTransformT",
    "LoRA",
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

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes + b.nbytes for a, b, _ in self._refs)


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
