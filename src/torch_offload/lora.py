"""LoRA types and per-weight merge / routed transforms.

:class:`LoRA` pairs, validates, and pins factor matrices from a flat
safetensors state dict at construction. The raw state dict is not retained —
the resource owns one immutable copy of the pinned factors that merge and
routed consumers may share.

Two application paths apply the resource's factors:

- :class:`LoRATransform` (merge mode) — applied to the GPU parameter
  after DMA; integrates with block streaming. Uses dense in-place
  ``addmm_`` when available, otherwise an adapter-provided
  dequantize/requantize plus ``copy_into`` path.
- routed mode (:func:`install_routed_residual_hook`) — a forward-PRE hook
  copies the target's factors from pinned CPU storage to the input device for
  that invocation; a forward-POST hook adds
  ``strength * (x @ A.T) @ B.T`` to the layer's output and drops those device
  copies. The base weight is not touched in place. Restricted to ``nn.Linear``
  parents (other layer types raise); shared weight storage is allowed (the
  hook targets the matched module, not the weight bytes). Factors are cast to
  the layer's output dtype before the residual, so quantized bases work as
  long as the matched module exposes a compatible logical ``nn.Linear``
  shape. Formats whose logical shape differs from their packed storage shape
  still need a richer per-format LoRA layer.

:class:`~torch_offload.ModelOffloader` is the consumer-facing API; its
``activate(..., loras=..., lora_mode=...)`` receives the requested path once
the device is known. The merge path runs
:class:`LoRATransform` from an activation-scoped post-copy hook; the
routed path lives as forward hooks installed on activate and removed
on deactivate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import nn

from .pinned_param import PinnedParam
from .tensor_adapter_registry import param_representation, select_adapter
from .tensor_adapters import (
    DenseAddmmTensorAdapter,
    DequantRequantCopyIntoTensorAdapter,
    adapter_name,
)

__all__ = [
    "LoRA",
    "LoRAFactor",
    "LoRAMode",
    "LoRATransform",
    "ScaledLoRAFactor",
]

LoRAMode = Literal["merge", "routed"]


@dataclass(slots=True, frozen=True)
class LoRAFactor:
    """A LoRA's pinned factor pair for one target weight.

    ``a`` is the ``(rank, in_dim)`` down-projection and ``b`` the
    ``(out_dim, rank)`` up-projection, each held as a :class:`PinnedParam`.
    Strength is *not* part of the pair — it is extrinsic and supplied when the
    LoRA is bound to a target. Per-pair shape
    validity is checked before pinning (in :func:`_validate_factor_pair`); the
    match against a concrete target shape is checked separately, where the
    target is known.

    :meth:`scaled` binds the extrinsic strength without discarding the pinned
    representation, so each application path can materialize the factors
    through their tensor adapters.
    """

    a: PinnedParam
    b: PinnedParam

    @property
    def cache_bytes(self) -> int:
        """Pinned host bytes held by this factor pair."""
        return self.a.cache_bytes + self.b.cache_bytes

    def scaled(self, strength: float) -> ScaledLoRAFactor:
        """Bind this pinned factor pair to ``strength``."""
        return ScaledLoRAFactor(self.a, self.b, strength)


@dataclass(slots=True, frozen=True)
class ScaledLoRAFactor:
    """A pinned factor pair bound to an application ``strength``.

    The application-side carrier used by :class:`LoRATransform` and routed
    hooks. Keeping :class:`PinnedParam` rather than CPU tensor views preserves
    adapter-specific reconstruction metadata such as a ``DTensor``'s original
    device mesh. The contribution to the base weight is
    ``strength * (b @ a)``.

    Use :meth:`from_tensors` when constructing a standalone transform from
    unpinned tensors. LoRA resources normally create this through
    :meth:`LoRAFactor.scaled` and reuse their existing pinned backing.
    """

    a: PinnedParam
    b: PinnedParam
    strength: float

    def __post_init__(self) -> None:
        if (
            len(self.a.shape) != 2
            or len(self.b.shape) != 2
            or self.a.shape[0] != self.b.shape[1]
        ):
            raise ValueError(
                f"LoRA factor shape mismatch: A shape is {tuple(self.a.shape)}, "
                f"B shape is {tuple(self.b.shape)}."
            )

    @classmethod
    def from_tensors(
        cls,
        a: torch.Tensor,
        b: torch.Tensor,
        strength: float,
    ) -> ScaledLoRAFactor:
        """Pin an unbound tensor pair and bind it to ``strength``."""
        return cls(
            PinnedParam(nn.Parameter(a, requires_grad=False)),
            PinnedParam(nn.Parameter(b, requires_grad=False)),
            strength,
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


class LoRA:
    """Reusable immutable pinned LoRA resource.

    Build once from a flat ``state_dict``: factor pairs are validated, cast to
    the optional storage ``dtype``, and pinned directly. The resource owns the
    single pinned copy and does not retain the raw ``state_dict``.

    Satisfies :class:`~torch_offload.protocols.ResourceStore`, so it can be
    registered in :class:`~torch_offload.ResourceCache` for budget tracking and
    policy-driven eviction. Merge and routed consumers read the same immutable
    factor backing and may overlap.

    Strength is extrinsic — specify it when passing the resource to
    :meth:`ModelOffloader.activate` via ``lora_strengths``.

    ``state_dict`` keys must already be model parameter paths (``.lora_A`` /
    ``.lora_B`` suffixed). Any key remapping — e.g. stripping the
    ``diffusion_model.`` prefix on ComfyUI adapters — is the caller's
    responsibility, done in the factory that produces the state dict.
    """

    def __init__(self, targets: Mapping[str, LoRAFactor]) -> None:
        self._targets = MappingProxyType(dict(targets))
        self._cache_bytes = sum(
            factor.cache_bytes for factor in self._targets.values()
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        dtype: torch.dtype | None = None,
    ) -> LoRA:
        """Pair, validate, build, and pin ``state_dict`` into a LoRA.

        ``dtype`` casts every factor before pinning. For routed mode, matching
        the model's compute dtype reduces pinned storage and per-forward H2D
        traffic. Left as ``None``, factors keep their stored dtype. Merge mode
        casts at apply time regardless.
        """
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError(f"LoRA dtype must be floating-point, got {dtype}.")
        _validate_lora_state_dict(state_dict)
        return cls(_pin_lora_targets(state_dict, dtype=dtype))

    @property
    def targets(self) -> Mapping[str, LoRAFactor]:
        """Immutable target-weight to pinned-factor mapping."""
        return self._targets

    @property
    def cache_bytes(self) -> int:
        return self._cache_bytes


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
            a = param_representation(factor.a.make_cpu_param())
            b = param_representation(factor.b.make_cpu_param())
            a_gpu = a.to(device=dev, dtype=dt, non_blocking=True)
            b_gpu = b.to(device=dev, dtype=dt, non_blocking=True)
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
                "LoRA factor shape mismatch: B@A produces "
                f"{factor.produced_shape}, target shape is {target_shape}."
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


@dataclass(slots=True, frozen=True)
class _StagedLoRAFactor:
    """Adapter-materialized factor pair owned for one forward invocation."""

    a: nn.Parameter
    b: nn.Parameter
    strength: float


def _routed_residual(
    x: torch.Tensor,
    factors: Sequence[_StagedLoRAFactor],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Routed contribution ``Σ strength_i · (x @ A_i.T) @ B_i.T``.

    Strength scales the intermediate ``M·r`` projection (cheaper than scaling
    the ``M·out`` result, and keeps it extrinsic to the stored factors rather
    than baked into a buffer).
    """
    x_compute = x.to(dtype=output_dtype)
    total: torch.Tensor | None = None
    for factor in factors:
        # The PRE hook has already reconstructed the factors on their proper
        # device representation. A dtype-only cast preserves wrappers such as
        # DTensor and their device meshes.
        a = param_representation(factor.a).to(dtype=output_dtype)
        b = param_representation(factor.b).to(dtype=output_dtype)
        part = ((x_compute @ a.T) * factor.strength) @ b.T
        total = part if total is None else total + part
    if total is None:
        raise ValueError("Routed LoRA residual requires at least one factor")
    return total


def _linear_input(
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> torch.Tensor:
    if inputs:
        x = inputs[0]
    else:
        try:
            x = kwargs["input"]
        except KeyError as exc:
            raise TypeError(
                "Routed LoRA expected the Linear input as either the first "
                "positional argument or the 'input' keyword"
            ) from exc
    if not isinstance(x, torch.Tensor):
        raise TypeError(
            "Routed LoRA requires the Linear input to be a torch.Tensor; "
            f"got {type(x).__name__}"
        )
    return x


def _stage_routed_factors(
    factors: Sequence[ScaledLoRAFactor],
    x: torch.Tensor,
) -> tuple[_StagedLoRAFactor, ...]:
    """Adapter-materialize pinned factors on the invocation's input device."""
    return tuple(
        _StagedLoRAFactor(
            factor.a.materialize(x.device, non_blocking=True),
            factor.b.materialize(x.device, non_blocking=True),
            factor.strength,
        )
        for factor in factors
    )


class _RoutedLoRAHookHandle:
    """Paired PRE/POST hook handle with per-invocation staged factors."""

    __slots__ = ("_post_handle", "_pre_handle", "_staged")

    def __init__(
        self,
        parent: nn.Module,
        factors: Sequence[ScaledLoRAFactor],
    ) -> None:
        self._staged: list[tuple[_StagedLoRAFactor, ...]] = []

        def pre_hook(
            _module: nn.Module,
            inputs: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            x = _linear_input(inputs, kwargs)
            self._staged.append(_stage_routed_factors(factors, x))

        def post_hook(
            _module: nn.Module,
            inputs: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: object,
        ) -> object:
            # ``always_call=True`` also reaches this hook when the Linear or an
            # earlier hook raises. In that case there may be no staged entry or
            # Tensor output; only discard any completed staging work.
            staged = self._staged.pop() if self._staged else None
            if staged is None or not isinstance(output, torch.Tensor):
                return output
            x = _linear_input(inputs, kwargs)
            return output + _routed_residual(x, staged, output.dtype)

        self._pre_handle = parent.register_forward_pre_hook(
            pre_hook,
            with_kwargs=True,
        )
        try:
            self._post_handle = parent.register_forward_hook(
                post_hook,
                with_kwargs=True,
                always_call=True,
            )
        except BaseException:
            self._pre_handle.remove()
            raise

    def remove(self) -> None:
        self._post_handle.remove()
        self._pre_handle.remove()
        self._staged.clear()


def install_routed_residual_hook(
    parent: nn.Module,
    factors: Sequence[ScaledLoRAFactor],
) -> _RoutedLoRAHookHandle:
    """Stage pinned factors in a PRE hook and add their residual in POST.

    One paired handle covers every LoRA targeting this parent. Device copies
    are scoped to a single invocation and released after the residual is
    enqueued, so routed mode needs no adapter activation lifecycle or block
    scheduler.
    """
    if not factors:
        raise ValueError("Routed LoRA hook requires at least one factor")
    return _RoutedLoRAHookHandle(parent, factors)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _pin_lora_targets(
    state_dict: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> Mapping[str, LoRAFactor]:
    """Pin each validated factor pair without building a module hierarchy."""
    a_tensors, b_tensors = _split_factor_tensors(state_dict)
    factors: dict[str, LoRAFactor] = {}
    for base, a_source in a_tensors.items():
        b_source = b_tensors[base]
        a_tensor = a_source if dtype is None else a_source.to(dtype=dtype)
        b_tensor = b_source if dtype is None else b_source.to(dtype=dtype)
        factors[f"{base}.weight"] = LoRAFactor(
            a=PinnedParam(nn.Parameter(a_tensor, requires_grad=False)),
            b=PinnedParam(nn.Parameter(b_tensor, requires_grad=False)),
        )
    return factors


def _validate_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    """Check ``state_dict`` is a well-formed LoRA before it is built.

    Every target needs a paired ``lora_A`` / ``lora_B`` and each factor must be
    a 2-D floating-point matrix with a matching inner rank.
    """
    a_tensors, b_tensors = _split_factor_tensors(state_dict)

    if not a_tensors and not b_tensors:
        raise ValueError("LoRA state_dict contains no factor pairs")

    a_only = set(a_tensors) - set(b_tensors)
    b_only = set(b_tensors) - set(a_tensors)
    if a_only or b_only:
        raise ValueError(
            f"Unpaired LoRA factors: A-only={sorted(a_only)}, "
            f"B-only={sorted(b_only)}. Each target needs both "
            f".lora_A.weight and .lora_B.weight."
        )

    for base_key, a in a_tensors.items():
        _validate_factor_pair(f"{base_key}.weight", a, b_tensors[base_key])


def _split_factor_tensors(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    a_tensors: dict[str, torch.Tensor] = {}
    b_tensors: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.endswith(".lora_A.weight"):
            a_tensors[key[: -len(".lora_A.weight")]] = tensor
        elif key.endswith(".lora_B.weight"):
            b_tensors[key[: -len(".lora_B.weight")]] = tensor
    return a_tensors, b_tensors


def _validate_factor_pair(
    target_key: str,
    a: torch.Tensor,
    b: torch.Tensor,
) -> None:
    if not a.is_floating_point() or not b.is_floating_point():
        raise ValueError(
            f"LoRA factors for {target_key!r}: must be floating-point; "
            f"got A.dtype={a.dtype}, B.dtype={b.dtype}."
        )
    if a.dim() != 2 or b.dim() != 2 or a.shape[0] != b.shape[1]:
        raise ValueError(
            f"LoRA factor shape mismatch for {target_key!r}: "
            f"A.shape={tuple(a.shape)}, B.shape={tuple(b.shape)}. "
            f"Expected A=(rank, in_dim), B=(out_dim, rank) with "
            f"A.shape[0] == B.shape[1]."
        )
