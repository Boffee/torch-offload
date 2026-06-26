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

from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import torch
from torch import nn

from .composite_component import CompositeComponentStore
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
    "LoRARouteHandle",
    "LoRATransform",
    "ScaledLoRAFactor",
]


@dataclass(slots=True, frozen=True)
class LoRAFactor:
    """A LoRA's pinned factor pair for one target weight.

    ``a`` is the ``(rank, in_dim)`` down-projection and ``b`` the
    ``(out_dim, rank)`` up-projection, each held as a :class:`PinnedParam`
    so the offload stack can stream them to GPU through the same target
    pool the model's weights use. Strength is *not* part of the pair — it is
    extrinsic and supplied when the LoRA is bound to a target. Per-pair shape
    validity is checked before pinning (in :func:`_validate_factor_pair`); the
    match against a concrete target shape is checked separately, where the
    target is known.

    The merge / current-routed paths consume *tensors*, so :meth:`scaled`
    materializes a device-ready :class:`ScaledLoRAFactor` over zero-copy
    views of the pinned host bytes; factor streaming instead consumes the
    :class:`PinnedParam`\\ s directly.
    """

    a: PinnedParam
    b: PinnedParam

    @property
    def cache_bytes(self) -> int:
        """Pinned host bytes held by this factor pair."""
        return self.a.cache_bytes + self.b.cache_bytes

    def scaled(self, strength: float) -> ScaledLoRAFactor:
        """Materialize a device-ready compute factor bound to ``strength``.

        Returns a :class:`ScaledLoRAFactor` over the pinned CPU tensors
        (zero-copy views of this factor's pinned host bytes). Callers move it
        to the target device / dtype at the apply site, so the pinned source
        keeps ``non_blocking`` host-to-device copies asynchronous.
        """
        return ScaledLoRAFactor(
            self.a.make_cpu_param().data,
            self.b.make_cpu_param().data,
            strength,
        )


@dataclass(slots=True, frozen=True)
class ScaledLoRAFactor:
    """A factor pair as plain tensors bound to an application ``strength``.

    The compute-side carrier eaten by :func:`_routed_residual` and
    :class:`LoRATransform`. Construct it directly from device/CPU tensors, or
    via :meth:`LoRAFactor.scaled` from a LoRA's pinned storage. The
    contribution to the base weight is ``strength * (b @ a)``.
    """

    a: torch.Tensor
    b: torch.Tensor
    strength: float

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


class LoRA:
    """A LoRA adapter whose factors are an ``nn.Module`` pinned via the
    offload substrate.

    At construction the flat ``state_dict`` is paired, validated, and built
    into an ``nn.Module`` whose factors sit at their dotted paths (numeric
    segments become ``nn.ModuleList`` positions, so a block-indexed LoRA lands
    each factor at its true block index); a
    :class:`~torch_offload.composite_component.CompositeComponentStore` then
    pins it — the same path the model itself uses. The store owns the single
    pinned copy, the raw ``state_dict`` is not retained, and :attr:`targets`
    is derived from it. Pass ``blocks_attr`` to group factor blocks as a
    streamed component (for routed co-scheduling with the model); the default
    pins the whole adapter.

    Implements both the :class:`~torch_offload.protocols.ResourceStore`
    and :class:`~torch_offload.protocols.ResourceBinding` shapes so it
    can be registered in :class:`~torch_offload.ModelCache` for budget
    tracking and policy-driven eviction. ``bind()`` returns ``self``;
    ``activate``/``deactivate`` are no-ops because factors stay on
    pinned CPU and are copied to GPU per-parameter by
    :class:`LoRATransform` during the merge-mode post-copy hook.

    Strength is extrinsic — specify it when passing the adapter to
    :meth:`ModelOffloader.set_loras` via the ``strengths`` argument.

    ``state_dict`` keys must already be model parameter paths (``.lora_A`` /
    ``.lora_B`` suffixed). Any key remapping — e.g. stripping the
    ``diffusion_model.`` prefix on ComfyUI adapters — is the caller's
    responsibility, done in the factory that produces the state dict.
    """

    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        blocks_attr: list[str] = [],  # noqa: B006  (read-only; never mutated)
    ) -> None:
        _validate_lora_state_dict(state_dict)
        self._module = _build_lora_module(state_dict)
        self._store = CompositeComponentStore.from_module(
            self._module, blocks_attr=blocks_attr,
        )

    @property
    def targets(self) -> dict[str, LoRAFactor]:
        """Map of target weight name to its pinned :class:`LoRAFactor`.

        Derived on demand from the composite store, which owns the single
        pinned copy of every factor. ``LoRAFactor.scaled()`` still views those
        pinned host bytes, so merge's ``non_blocking`` H2D stays asynchronous.
        """
        return _derive_targets(self._store)

    @property
    def cache_bytes(self) -> int:
        return self._store.cache_bytes

    @property
    def value(self) -> LoRA:
        return self

    def bind(self) -> LoRA:
        return self

    def activate(
        self, device: torch.device | str | None = None, **kwargs: object,
    ) -> None:
        del device, kwargs

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


def _routed_residual(
    x: torch.Tensor,
    factors: Sequence[ScaledLoRAFactor],
) -> torch.Tensor:
    """Routed LoRA contribution ``Σ_i strength_i · (x @ A_i.T) @ B_i.T``.

    Each adapter pair is applied independently and summed — no fusion.
    Strength scales the intermediate ``M·r`` projection (cheaper than
    scaling the ``M·out`` result, and keeps it extrinsic to the stored
    factors rather than baked into a buffer). The overwhelmingly common
    single-adapter case is one scaled pair of GEMMs.
    """
    total: torch.Tensor | None = None
    for factor in factors:
        part = ((x @ factor.a.T) * factor.strength) @ factor.b.T
        total = part if total is None else total + part
    if total is None:
        raise ValueError("Routed LoRA residual requires at least one factor.")
    return total


class LoRARouteHandle:
    """Live forward-hook for one routed LoRA target.

    Owns GPU copies of the LoRA factors plus the registered hook on
    the parent module. Forward path becomes::

        y = base(x) + sum_i strength_i * (x @ A_i.T) @ B_i.T

    Each adapter's ``(A, B)`` pair is applied independently and summed
    (see :func:`_routed_residual`); strength is applied at the
    intermediate ``M·r`` projection, never baked into a stored buffer.
    The overwhelmingly common single-adapter case is one scaled pair of
    GEMMs; multiple adapters against one weight (rare) do N small GEMMs.
    Keeping the pairs separate makes each factor a plain tensor the mover
    handles directly and keeps the path DTensor-friendly — no fused
    buffer that would discard tensor-parallel placement.

    The factors are copied pinned-CPU → GPU once at install (``to`` casts
    dtype during the transfer, so fp32-stored factors land as the layer's
    bf16/fp16). Construction installs the hook; :meth:`remove` removes it
    and drops the GPU factor refs so refcount-GC reclaims them.

    Restricted to ``nn.Linear``-shaped forwards. The math assumes
    ``base(x) = x @ W.T (+ bias)``; LoRA applied to Conv2d, Embedding,
    or other layouts needs different formulas (see PEFT's per-type
    LoraLayer subclasses).
    """

    __slots__ = ("_factors", "_handle")

    def __init__(
        self,
        parent: nn.Module,
        factors: Sequence[ScaledLoRAFactor],
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> None:
        # Move each factor pair to the target device/dtype once; strength
        # stays extrinsic and is applied at the hook, not baked in.
        gpu_factors = [
            ScaledLoRAFactor(
                factor.a.to(device=device, dtype=dtype, non_blocking=True),
                factor.b.to(device=device, dtype=dtype, non_blocking=True),
                factor.strength,
            )
            for factor in factors
        ]
        # The closure captures `gpu_factors` so the GPU tensors stay alive
        # while the hook is registered; self._factors holds the same refs
        # so remove() can drop them.
        self._factors = gpu_factors

        def hook(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            return output + _routed_residual(inputs[0], gpu_factors)

        self._handle = parent.register_forward_hook(hook)

    def remove(self) -> None:
        self._handle.remove()
        # Drop the GPU factors. Unregistering removes the hook function from
        # the module's _forward_hooks dict, so the closure that also captured
        # them becomes unreachable and Python refcount-GCs it.
        self._factors = None


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _build_lora_module(state_dict: dict[str, torch.Tensor]) -> nn.Module:
    """Build an ``nn.Module`` from a LoRA state dict.

    The state-dict keys are already module paths, so each factor tensor is
    placed directly at its path — no pairing or target synthesis. Only
    ``.lora_A.weight`` / ``.lora_B.weight`` entries are kept; non-factor
    entries (e.g. ``.alpha`` scalars) are ignored. A numeric path segment
    indexes an ``nn.ModuleList`` (extended with empty holder modules to reach
    the index, so block positions stay true even when some blocks are
    unadapted); a name segment is an attribute submodule. Well-formedness is
    checked separately by :func:`_validate_lora_state_dict`.
    """
    root = nn.Module()
    for key, tensor in state_dict.items():
        if not key.endswith((".lora_A.weight", ".lora_B.weight")):
            continue
        *mod_segs, param_name = key.split(".")
        node: nn.Module = root
        for i, seg in enumerate(mod_segs):
            # A holder is an nn.ModuleList exactly when its own child is a
            # numeric index; this is what lets ModuleLists nest (``0.1.``).
            child_is_index = i + 1 < len(mod_segs) and mod_segs[i + 1].isdigit()
            if seg.isdigit():
                # The parent of a numeric segment was created as a ModuleList
                # (its own child_is_index was True). A failure here means the
                # state dict uses one path as both a submodule and a list
                # index — a malformed adapter we reject rather than mis-build.
                assert isinstance(node, nn.ModuleList)
                idx = int(seg)
                while len(node) <= idx:
                    node.append(nn.ModuleList() if child_is_index else nn.Module())
                node = node[idx]
            else:
                child = getattr(node, seg, None)
                if child is None:
                    child = nn.ModuleList() if child_is_index else nn.Module()
                    node.add_module(seg, child)
                node = child
        node.register_parameter(
            param_name, nn.Parameter(tensor, requires_grad=False),
        )
    return root


def _validate_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    """Check ``state_dict`` is a well-formed LoRA before it is built.

    Every target needs a paired ``lora_A`` / ``lora_B`` and each factor must be
    a 2-D floating-point matrix with a matching inner rank.
    """
    a_tensors, b_tensors = _split_factor_tensors(state_dict)

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
    target_key: str, a: torch.Tensor, b: torch.Tensor,
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


def _derive_targets(store: CompositeComponentStore) -> dict[str, LoRAFactor]:
    """Recover the ``target -> LoRAFactor`` map from the pinned store.

    The store owns the single pinned copy of every factor; we pair its
    ``.lora_A`` / ``.lora_B`` pinned params back into :class:`LoRAFactor`s.
    """
    pinned = store.pinned_params()
    a_by_base: dict[str, PinnedParam] = {}
    b_by_base: dict[str, PinnedParam] = {}
    for name, param in pinned.items():
        if name.endswith(".lora_A.weight"):
            a_by_base[name[: -len(".lora_A.weight")]] = param
        elif name.endswith(".lora_B.weight"):
            b_by_base[name[: -len(".lora_B.weight")]] = param

    factors: dict[str, LoRAFactor] = {}
    for base, a in a_by_base.items():
        factors[f"{base}.weight"] = LoRAFactor(a=a, b=b_by_base[base])
    return factors
