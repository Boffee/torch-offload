"""LoRA types and per-weight merge / routed transforms.

:class:`LoRA` pairs, validates, and pins factor matrices from a flat
safetensors state dict at construction.  The raw state dict is not
retained — this object owns the only copy of the pinned factors.

Two application paths share the same :class:`LoRA` data container:

- :class:`LoRATransform` (merge mode) — applied to the GPU parameter
  after DMA; integrates with block streaming. Uses dense in-place
  ``addmm_`` when available, otherwise an adapter-provided
  dequantize/requantize plus ``copy_into`` path.
- :class:`LoRARouteHandle` (routed mode) — installs a forward hook on
  the layer that adds ``alpha * (x @ A.T @ B.T)`` to the layer's output;
  base weight is not touched in place. Restricted to ``nn.Linear``
  parents (other layer types raise) and tied weights are rejected.
  Compatible with quantized bases whose adapter can report the logical
  compute dtype, or that expose ``module.compute_dtype``. Formats whose
  logical shape does not match their packed storage shape still need a
  richer per-format LoRA layer.

:class:`~torch_offload.ModelOffloader` is the consumer-facing API; its
``set_loras(..., mode=...)`` records the requested path and activation
applies it once the device is known. The merge path runs
:class:`LoRATransform` from an activation-scoped post-copy hook; the
routed path lives as forward hooks installed on activate and removed
on deactivate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import torch
from torch import nn

from .tensor_adapter_registry import param_representation, select_adapter
from .tensor_adapters import (
    DenseAddmmTensorAdapter,
    DequantRequantCopyIntoTensorAdapter,
    adapter_name,
)

__all__ = [
    "FusedLoRAFactors",
    "KeyTransformT",
    "LoRA",
    "LoRAFactor",
    "LoRARouteHandle",
    "LoRATransform",
    "ScaledLoRAFactor",
    "default_key_transform",
]


KeyTransformT = Callable[[str], str] | None


@dataclass(slots=True, frozen=True)
class LoRAFactor:
    """A LoRA's pinned factor pair for one target weight.

    ``a`` is the ``(rank, in_dim)`` down-projection and ``b`` the
    ``(out_dim, rank)`` up-projection. Strength is *not* part of the pair —
    it is extrinsic and supplied when the LoRA is bound to a target (see
    :meth:`scaled` / :class:`ScaledLoRAFactor`). Per-pair shape validity is
    checked at construction; the match against a concrete target shape is
    checked separately, where the target is known.
    """

    a: torch.Tensor
    b: torch.Tensor

    def __post_init__(self) -> None:
        if self.a.ndim != 2 or self.b.ndim != 2 or self.a.shape[0] != self.b.shape[1]:
            raise ValueError(
                f"LoRA factor shape mismatch: A shape is {tuple(self.a.shape)}, B shape is {tuple(self.b.shape)}."
            )

    @property
    def rank(self) -> int:
        return self.a.shape[0]

    @property
    def in_dim(self) -> int:
        return self.a.shape[1]

    @property
    def out_dim(self) -> int:
        return self.b.shape[0]

    @property
    def produced_shape(self) -> tuple[int, int]:
        """Shape of ``b @ a`` — the base-weight shape this factor targets."""
        return (self.b.shape[0], self.a.shape[1])

    def scaled(self, strength: float) -> ScaledLoRAFactor:
        """Bind this factor pair to an application ``strength``."""
        return ScaledLoRAFactor(self.a, self.b, strength)


@dataclass(slots=True, frozen=True)
class ScaledLoRAFactor(LoRAFactor):
    """A :class:`LoRAFactor` bound to the ``strength`` it is applied at.

    The contribution to the base weight is ``strength * (b @ a)``; strength
    is fixed when the LoRA is bound to a target (e.g. via ``set_loras``).
    """

    strength: float


@dataclass(slots=True, frozen=True)
class FusedLoRAFactors:
    """Several :class:`ScaledLoRAFactor` against one target, fused into a
    single rank-stacked pair on a target device.

    ``a_fused`` has shape ``(sum_i r_i, in_dim)`` — each factor's ``a``
    stacked along the rank axis. ``b_fused`` has shape
    ``(out_dim, sum_i r_i)`` — each factor's ``b`` scaled by its strength,
    stacked along the rank axis. The identity
    ``b_fused @ a_fused == sum_i strength_i * (b_i @ a_i)`` falls out of the
    block structure: every fused-rank column belongs to exactly one factor's
    slice, so there are no cross terms. Mixed ranks are handled naturally —
    the fused rank is ``sum_i r_i`` rather than ``N * r``.

    This is a **per-target** fusion (several adapters against one base
    weight); it is not a cross-block batching mechanism.
    """

    a_fused: torch.Tensor
    b_fused: torch.Tensor

    @classmethod
    def from_factors(
        cls,
        factors: Sequence[ScaledLoRAFactor],
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> FusedLoRAFactors:
        """Stack ``factors`` into one fused pair on ``device``.

        Each factor's pinned ``a``/``b`` is DMA'd straight into its rank
        slice (``copy_`` casts dtype during the transfer, so fp32-stored
        factors land as the layer's bf16/fp16 in one op), and ``strength``
        is baked into ``b`` in place — no intermediate scaled-``b`` tensor.
        All factors must share ``in_dim``/``out_dim`` (they parameterize the
        same base weight); this is validated defensively.
        """
        if not factors:
            raise ValueError("FusedLoRAFactors.from_factors requires at least one ScaledLoRAFactor.")
        in_dim = factors[0].in_dim
        out_dim = factors[0].out_dim
        for factor in factors:
            if factor.in_dim != in_dim or factor.out_dim != out_dim:
                raise ValueError(
                    f"LoRA factor shape mismatch: expected A=(r, {in_dim}), "
                    f"B=({out_dim}, r); got A={tuple(factor.a.shape)}, "
                    f"B={tuple(factor.b.shape)}."
                )
        total_rank = sum(factor.rank for factor in factors)
        factor_dtype = dtype if dtype is not None else factors[0].a.dtype

        a_fused = torch.empty(
            (total_rank, in_dim),
            device=device,
            dtype=factor_dtype,
        )
        b_fused = torch.empty(
            (out_dim, total_rank),
            device=device,
            dtype=factor_dtype,
        )
        offset = 0
        for factor in factors:
            r = factor.rank
            a_fused[offset : offset + r].copy_(factor.a, non_blocking=True)
            b_slice = b_fused[:, offset : offset + r]
            b_slice.copy_(factor.b, non_blocking=True)
            b_slice.mul_(factor.strength)
            offset += r

        return cls(a_fused=a_fused, b_fused=b_fused)


def default_key_transform(key: str) -> str:
    """Strip the common ``diffusion_model.`` prefix from ComfyUI LoRA keys."""
    prefix = "diffusion_model."
    return key[len(prefix) :] if key.startswith(prefix) else key


class LoRA:
    """A LoRA adapter with pinned factor matrices.

    Factors are paired, validated, and pinned to host memory at
    construction.  The raw ``state_dict`` is not retained.

    Implements both the :class:`~torch_offload.protocols.ResourceStore`
    and :class:`~torch_offload.protocols.ResourceBinding` shapes so it
    can be registered in :class:`~torch_offload.ModelCache` for budget
    tracking and policy-driven eviction. ``bind()`` returns ``self``;
    ``activate``/``deactivate`` are no-ops because factors stay on
    pinned CPU and are copied to GPU per-parameter by
    :class:`LoRATransform` during the merge-mode post-copy hook.

    Strength is extrinsic — specify it when passing the adapter to
    :meth:`ModelOffloader.set_loras` via the ``strengths`` argument.

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
    def targets(self) -> dict[str, LoRAFactor]:
        """Map of target weight name to its pinned :class:`LoRAFactor`."""
        return self._factors

    @property
    def cache_bytes(self) -> int:
        return sum(factor.a.nbytes + factor.b.nbytes for factor in self._factors.values())

    @property
    def value(self) -> LoRA:
        return self

    def bind(self) -> LoRA:
        return self

    def activate(self, device: torch.device | str | None = None) -> None:
        del device

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
    """Per-weight LoRA factors applied to one base parameter.

    Holds references to LoRA-owned pinned factor matrices — no cloning
    or pinning happens here. :meth:`apply` copies each factor pair to
    the target parameter's device and applies the update using either
    dense in-place ``addmm_`` or the tensor adapter's
    dequantize/requantize plus ``copy_into`` capability. The target
    :class:`~torch.nn.Parameter` object is always preserved.
    """

    __slots__ = ("_factors",)

    def __init__(self, factors: list[ScaledLoRAFactor]) -> None:
        self._factors = factors

    def validate_target(self, param: nn.Parameter) -> None:
        """Raise if ``param`` cannot receive this LoRA merge.

        This is an optional preflight for callers that want an earlier
        error. :meth:`apply` uses the same validation path immediately
        before mutating the target parameter.
        """
        representation = param_representation(param)
        adapter = _dequant_requant_adapter(representation)
        _validate_factor_shapes(
            self._factors,
            self._logical_shape(representation, adapter),
        )

    def apply(self, param: nn.Parameter) -> None:
        # Operate on the representation tensor: ``param.data`` for plain and
        # wrapped-quant parameters, but the param itself for a Parameter
        # subclass whose ``.data`` is lossy (bitsandbytes Params4bit).
        representation = param_representation(param)
        adapter = _dequant_requant_adapter(representation)
        if adapter is None:
            _validate_factor_shapes(self._factors, tuple(representation.shape))
            self._apply_dense(representation)
            return

        dense = adapter.dequantize(representation)
        # Validate against the dense LOGICAL shape, not the wrapper's shape:
        # Params4bit reports its packed (numel/2, 1) storage shape, so the
        # factor check must use the dequantized shape to be correct.
        _validate_factor_shapes(self._factors, tuple(dense.shape))
        self._apply_dense(dense)
        new_data = adapter.requantize(dense, like=representation)
        adapter.copy_into(new_data, target=representation)

    @staticmethod
    def _logical_shape(
        representation: torch.Tensor,
        adapter: DequantRequantCopyIntoTensorAdapter[Any, Any] | None,
    ) -> tuple[int, ...]:
        # Plain / dense-addmm targets carry their logical shape directly;
        # dequant/requant targets may wrap a different storage shape, so the
        # logical shape comes from a dequantized view.
        if adapter is None:
            return tuple(representation.shape)
        return tuple(adapter.dequantize(representation).shape)

    def _apply_dense(self, data: torch.Tensor) -> None:
        dev, dt = data.device, data.dtype
        for factor in self._factors:
            a_gpu = factor.a.to(device=dev, dtype=dt, non_blocking=True)
            b_gpu = factor.b.to(device=dev, dtype=dt, non_blocking=True)
            data.addmm_(b_gpu, a_gpu, alpha=factor.strength)


def _validate_factor_shapes(
    factors: Sequence[ScaledLoRAFactor],
    target_shape: tuple[int, ...],
) -> None:
    # Per-pair validity (2-D, matching inner rank) is guaranteed by
    # ScaledLoRAFactor construction; only the match against this concrete
    # target shape is checked here.
    for factor in factors:
        if factor.produced_shape != target_shape:
            raise ValueError(
                f"LoRA factor shape mismatch: B@A produces {factor.produced_shape}, target shape is {target_shape}."
            )


def _dequant_requant_adapter(
    data: torch.Tensor,
) -> DequantRequantCopyIntoTensorAdapter[Any, Any] | None:
    try:
        adapter = select_adapter(data)
    except NotImplementedError as exc:
        raise ValueError(
            f"Tensor type {type(data).__name__} has no registered tensor adapter. "
            "Merge requires a tensor adapter with dense addmm or "
            "dequantize/requantize plus copy_into support."
        ) from exc

    if isinstance(adapter, DenseAddmmTensorAdapter):
        try:
            adapter.validate_dense_addmm_target(data)
        except ValueError as exc:
            if isinstance(adapter, DequantRequantCopyIntoTensorAdapter):
                return adapter
            raise ValueError(str(exc)) from exc
        return None

    if isinstance(adapter, DequantRequantCopyIntoTensorAdapter):
        return adapter

    raise ValueError(
        f"{adapter_name(adapter)} does not support dense in-place addmm "
        "or dequantize/requantize plus copy_into updates. Use routed "
        "LoRA for this tensor type."
    )


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

    __slots__ = ("_fused", "_handle")

    def __init__(
        self,
        parent: nn.Module,
        factors: Sequence[ScaledLoRAFactor],
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> None:
        fused = FusedLoRAFactors.from_factors(
            factors,
            device=device,
            dtype=dtype,
        )
        # Bind the fused tensors as locals so the hook closure keeps them
        # alive; self._fused holds the same refs so remove() can drop them.
        a_fused = fused.a_fused
        b_fused = fused.b_fused
        self._fused = fused

        def hook(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            x = inputs[0]
            return output + (x @ a_fused.T) @ b_fused.T

        self._handle = parent.register_forward_hook(hook)

    def remove(self) -> None:
        self._handle.remove()
        # Drop the fused GPU factors. The closure also holds them via the
        # captured locals, but unregistering removes the hook function from
        # the module's _forward_hooks dict, so the closure becomes
        # unreachable and Python refcount-GCs it.
        self._fused = None


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _pair_and_pin(
    state_dict: dict[str, torch.Tensor],
    key_transform: KeyTransformT,
) -> dict[str, LoRAFactor]:
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

    factors: dict[str, LoRAFactor] = {}
    for base_key, a in a_tensors.items():
        b = b_tensors[base_key]
        target_key = f"{base_key}.weight"
        if key_transform is not None:
            target_key = key_transform(target_key)

        if not a.is_floating_point() or not b.is_floating_point():
            raise ValueError(
                f"LoRA factors for {target_key!r}: must be floating-point; got A.dtype={a.dtype}, B.dtype={b.dtype}."
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
                f"Duplicate LoRA target {target_key!r}: key_transform mapped multiple source keys to the same target."
            )
        factors[target_key] = LoRAFactor(
            a.contiguous().pin_memory(),
            b.contiguous().pin_memory(),
        )

    return factors
