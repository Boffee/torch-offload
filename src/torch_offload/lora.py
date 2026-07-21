"""LoRA types and per-weight merge / routed transforms.

:class:`LoRA` pairs, validates, and pins factor matrices from a flat
safetensors state dict at construction. The raw state dict is not retained —
the resource owns the only copy of the pinned factors plus one exclusive
activation lifecycle.

Two application paths apply the resource's factors:

- :class:`LoRATransform` (merge mode) — applied to the GPU parameter
  after DMA; integrates with block streaming. Uses dense in-place
  ``addmm_`` when available, otherwise an adapter-provided
  dequantize/requantize plus ``copy_into`` path.
- routed mode (:func:`install_routed_residual_hook`) — a forward-POST hook
  on the layer adds ``strength * (x @ A.T) @ B.T`` to the layer's output,
  reading the LoRA's GPU-resident factors (streamed co-scheduled with the
  base model via the :class:`LoRA` resource) off their holder modules; the
  base weight is not touched in place. Restricted to ``nn.Linear`` parents
  (other layer types raise); shared weight storage is allowed (the hook
  targets the matched module, not the weight bytes). The factor is cast to
  the layer's output dtype in the hook, so quantized bases work as long as
  the matched module exposes the logical ``nn.Linear`` weight shape. Formats
  whose logical shape does not match their packed storage shape still need a
  richer per-format LoRA layer.

:class:`~torch_offload.ModelOffloader` is the consumer-facing API; its
``activate(..., loras=..., lora_mode=...)`` receives the requested path once
the device is known. The merge path runs
:class:`LoRATransform` from an activation-scoped post-copy hook; the
routed path lives as forward hooks installed on activate and removed
on deactivate.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from ._devices import canonical_device
from .composite_component import CompositeComponent, CompositeComponentStore
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
    "LoRARuntimeInUseError",
    "LoRATransform",
    "LoraMode",
    "ScaledLoRAFactor",
]

LoraMode = Literal["merge", "routed"]


class LoRARuntimeInUseError(RuntimeError):
    """A LoRA resource already has an active use."""


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

    The merge path consumes *tensors*, so :meth:`scaled` materializes a
    device-ready :class:`ScaledLoRAFactor` over zero-copy views of the pinned
    host bytes; routed mode instead streams the :class:`PinnedParam`\\ s
    directly and reads each GPU-resident factor off its holder module.
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

    The compute-side carrier eaten by :class:`LoRATransform` (merge mode).
    Construct it directly from device/CPU tensors, or via
    :meth:`LoRAFactor.scaled` from a LoRA's pinned storage. The contribution
    to the base weight is ``strength * (b @ a)``.
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
    """Reusable pinned LoRA resource with one exclusive active use.

    Build once from a flat ``state_dict``: the factors are paired, validated,
    and built into an ``nn.Module`` whose factors sit at their dotted paths
    (numeric segments become ``nn.ModuleList`` positions, so a block-indexed
    LoRA lands each factor at its true block index); a
    :class:`~torch_offload.composite_component.CompositeComponentStore` then
    pins it — the same path the model itself uses. The resource owns the single
    pinned copy, the raw ``state_dict`` is not retained, and :attr:`targets`
    is derived from it. Pass ``blocks_attr`` to group factor blocks as
    streamed components (for routed co-scheduling with the model); the
    default pins the whole adapter.

    Satisfies :class:`~torch_offload.protocols.ResourceStore`, so it can be
    registered in :class:`~torch_offload.ResourceCache` for budget tracking and
    policy-driven eviction. The cached resource is reused sequentially: every
    merge or routed use must call :meth:`activate` and :meth:`deactivate`, and
    an overlapping activation fails immediately with
    :class:`LoRARuntimeInUseError`. Merge activation only holds that exclusive
    claim; routed activation additionally streams factors GPU-resident,
    co-scheduled with the base model.

    Strength is extrinsic — specify it when passing the resource to
    :meth:`ModelOffloader.activate` via ``lora_strengths``.

    ``state_dict`` keys must already be model parameter paths (``.lora_A`` /
    ``.lora_B`` suffixed). Any key remapping — e.g. stripping the
    ``diffusion_model.`` prefix on ComfyUI adapters — is the caller's
    responsibility, done in the factory that produces the state dict.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        composite_store: CompositeComponentStore,
    ) -> None:
        self._module = module
        self._composite_store = composite_store
        self._targets = _derive_targets(composite_store)
        self._activation_lock = threading.Lock()
        self._active = False
        self._composite: CompositeComponent | None = None

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        blocks_attr: Sequence[str] = (),
        dtype: torch.dtype | None = None,
    ) -> LoRA:
        """Pair, validate, build, and pin ``state_dict`` into a LoRA.

        ``dtype`` casts every factor at build time. For routed mode, pass the
        base model's compute dtype: the factors then stream and reside at that
        width and the forward hook's per-call cast is a no-op. Left as ``None``
        the factors keep their stored dtype (fp32 safetensors stay fp32 —
        resident at full width and re-cast on every forward). Merge mode casts
        at apply time regardless, so ``dtype`` only affects residency there.
        """
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError(f"LoRA dtype must be floating-point, got {dtype}.")
        _validate_lora_state_dict(state_dict)
        module = _build_lora_module(state_dict, dtype=dtype)
        composite_store = CompositeComponentStore.from_module(
            module,
            blocks_attr=blocks_attr,
        )
        return cls(module=module, composite_store=composite_store)

    def factor_holders(self, target_key: str) -> tuple[nn.Module, nn.Module]:
        """Resolve a target weight key to its ``(lora_A, lora_B)`` holder modules.

        Inverts the target convention (``<base>.weight`` is derived from the
        factor paths ``<base>.lora_A.weight`` / ``<base>.lora_B.weight``): the
        holders live at ``<base>.lora_A`` / ``<base>.lora_B`` in the owned
        factor tree. Routed mode reads their ``.weight`` fresh each forward —
        streaming swaps the Parameter on load/evict while the holder module
        identity is stable, so callers must hold the *module*, never the weight
        tensor.
        """
        base = target_key.removesuffix(".weight")
        return (
            self._module.get_submodule(f"{base}.lora_A"),
            self._module.get_submodule(f"{base}.lora_B"),
        )

    @property
    def targets(self) -> Mapping[str, LoRAFactor]:
        """Map of target weight name to its pinned :class:`LoRAFactor`.

        Derived once from the composite store and exposed as an immutable
        mapping. The store owns the single pinned copy of every factor —
        including the streamed block groups, whose pinned host masters are
        returned alongside the pinned remainder. ``LoRAFactor.scaled()`` still
        views those pinned host bytes, so merge's ``non_blocking`` H2D stays
        asynchronous.
        """
        return self._targets

    @property
    def cache_bytes(self) -> int:
        return self._composite_store.cache_bytes

    def activate(
        self,
        device: torch.device | str | None = None,
        *,
        mode: LoraMode,
        schedule_model: nn.Module | None = None,
        **kwargs: object,
    ) -> None:
        """Claim this LoRA for one merge or routed use.

        Merge mode keeps factors in pinned CPU storage and only holds the
        exclusive-use claim. Routed mode additionally binds a transient
        component to this LoRA's factor-holder module and co-schedules its
        streamed blocks against the required ``schedule_model``.
        """
        if mode not in ("merge", "routed"):
            raise ValueError(
                "LoRA mode must be 'merge' or 'routed', "
                f"got {mode!r}"
            )
        active_device: torch.device | None = None
        if mode == "routed":
            if device is None:
                raise ValueError("LoRA routed activation requires a device")
            if schedule_model is None:
                raise ValueError(
                    "LoRA routed activation requires a schedule_model"
                )
            active_device = canonical_device(device)

        if not self._activation_lock.acquire(blocking=False):
            raise LoRARuntimeInUseError(
                "LoRA already has an active use; overlapping LoRA "
                "activations are not supported"
            )

        self._active = True
        try:
            if mode == "routed":
                assert active_device is not None
                composite = self._composite_store.bind(
                    self._module,
                    schedule_model=schedule_model,
                )
                self._composite = composite
                composite.activate(active_device, **kwargs)
        except BaseException:
            try:
                if self._composite is not None:
                    self._composite.deactivate()
            finally:
                self._composite = None
                self._active = False
                self._activation_lock.release()
            raise

    def deactivate(self) -> None:
        """Release the active use and any routed factor residency."""
        if not self._active:
            return
        composite = self._composite
        self._composite = None
        try:
            if composite is not None:
                composite.deactivate()
        finally:
            self._active = False
            self._activation_lock.release()


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
    a: torch.Tensor,
    b: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """One routed LoRA pair's contribution: ``strength · (x @ A.T) @ B.T``.

    Strength scales the intermediate ``M·r`` projection (cheaper than scaling
    the ``M·out`` result, and keeps it extrinsic to the stored factors rather
    than baked into a buffer).
    """
    return ((x @ a.T) * strength) @ b.T


def install_routed_residual_hook(
    parent: nn.Module,
    lora_a: nn.Module,
    lora_b: nn.Module,
    strength: float,
) -> RemovableHandle:
    """Install a forward-POST hook on ``parent`` adding one LoRA's residual.

    ``lora_a`` / ``lora_b`` are the LoRA module's factor *holder* submodules.
    The hook reads their GPU-resident ``.weight`` **fresh on every forward**:
    streaming swaps the ``.weight`` Parameter on each load/evict, so the holder
    modules (stable identity) are captured, not their tensors. The factor is
    cast to the layer's output dtype in the hook, so the residual matches
    whatever the (possibly quantized) base layer produced.

    Multiple LoRAs on one ``parent`` stack as independent additive hooks —
    PyTorch chains them, so multi-LoRA summation falls out for free. Returns
    the removable handle; routed teardown removes it and deactivates the LoRA
    resource that owns the factors.
    """

    def hook(
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        a = lora_a.get_parameter("weight").to(output.dtype)
        b = lora_b.get_parameter("weight").to(output.dtype)
        return output + _routed_residual(inputs[0], a, b, strength)

    return parent.register_forward_hook(hook)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _build_lora_module(
    state_dict: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Build an ``nn.Module`` from a LoRA state dict.

    The state-dict keys are already module paths, so each factor tensor is
    placed directly at its path — no pairing or target synthesis. Only
    ``.lora_A.weight`` / ``.lora_B.weight`` entries are kept; non-factor
    entries (e.g. ``.alpha`` scalars) are ignored. A numeric path segment
    indexes an ``nn.ModuleList`` (extended with empty holder modules to reach
    the index, so block positions stay true even when some blocks are
    unadapted); a name segment is an attribute submodule. Well-formedness is
    checked separately by :func:`_validate_lora_state_dict`. ``dtype`` casts
    each factor as it is placed.
    """
    factor_keys = [key for key in state_dict if key.endswith((".lora_A.weight", ".lora_B.weight"))]
    # The root holder is itself an nn.ModuleList when paths start with an
    # index — a root-level nn.Sequential / nn.ModuleList model whose params
    # are ``0.weight`` etc., so factor keys are ``0.lora_A.weight``.
    root: nn.Module = nn.ModuleList() if any(key.split(".", 1)[0].isdigit() for key in factor_keys) else nn.Module()
    for key in factor_keys:
        tensor = state_dict[key]
        if dtype is not None:
            tensor = tensor.to(dtype)
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
            param_name,
            nn.Parameter(tensor, requires_grad=False),
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
    target_key: str,
    a: torch.Tensor,
    b: torch.Tensor,
) -> None:
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


def _derive_targets(
    store: CompositeComponentStore,
) -> Mapping[str, LoRAFactor]:
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
    return MappingProxyType(factors)
