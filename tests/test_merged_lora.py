"""Tests for activation-scoped LoRA application through ``ModelOffloader``.

Covers LoRA construction validation, activation matching, lifecycle
(activate/deactivate), LoRA switching, and forward-output correctness
against a manually-merged baseline.

Most lifecycle tests run on CPU (the merge math is device-agnostic);
CUDA-only tests gate on availability.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from torch_offload import (
    CachedModelRunner,
    LoRA,
    LoRAFactor,
    LoRAMode,
    LoRARuntimeInUseError,
    LoRATransform,
    ResourceCache,
    ModelOffloader,
    ModelSpec,
    ResourceNotRegisteredError,
    ResourceTooLargeError,
    PinnedComponent,
    LoRASpec,
    ScaledLoRAFactor,
    StreamConfig,
    StreamedComponent,
    merge_lora,
)
from torch_offload.pinned_module import PinnedModuleInstance
from torch_offload.pinned_param import PinnedParam
from torch_offload.protocols import (
    ResourceBinding,
    ResourceStore,
)

from tests.conftest import activated_model, streamed_components

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _factor_tensors(factor: LoRAFactor) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize a pinned :class:`LoRAFactor`'s ``(a, b)`` as CPU tensors."""
    return factor.a.make_cpu_param().data, factor.b.make_cpu_param().data


def _assert_lora_routing_available(lora: LoRA) -> None:
    """Prove that ``lora`` no longer holds its routed-use claim."""
    assert not lora._active
    assert lora._activation_lock.acquire(blocking=False)
    lora._activation_lock.release()


def _make_model_offloader(
    model: nn.Module,
    *,
    blocks_attr: list[str] = [],
    stream_trainable_weights: bool = False,
) -> ModelOffloader:
    return ModelOffloader.from_module(
        model,
        blocks_attr=blocks_attr,
        stream_trainable_weights=stream_trainable_weights,
    )


def _make_bf16_model(num_blocks: int = 4, dim: int = 16) -> nn.Module:
    """Tiny block-streaming-shaped model with bf16 frozen params."""

    class Block(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.attn = nn.Linear(dim, dim, bias=False)
            self.ff = nn.Linear(dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.ff(self.attn(x))

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(dim, dim, bias=False)
            self.transformer_blocks = nn.ModuleList([Block(dim) for _ in range(num_blocks)])
            self.head = nn.Linear(dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            for blk in self.transformer_blocks:
                x = blk(x)
            return self.head(x)

    m = M()
    m = m.to(torch.bfloat16)
    for p in m.parameters():
        p.requires_grad = False
    return m


def _make_tied_non_block_model(
    num_blocks: int = 2,
    dim: int = 16,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    class Block(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.attn = nn.Linear(dim, dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.attn(x)

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(dim, dim, bias=False)
            self.transformer_blocks = nn.ModuleList([Block(dim) for _ in range(num_blocks)])
            self.head = nn.Linear(dim, dim, bias=False)
            self.head.weight = self.embed.weight

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            for blk in self.transformer_blocks:
                x = blk(x)
            return self.head(x)

    model = M().to(dtype)
    for p in model.parameters():
        p.requires_grad = False
    return model


def _make_lora_sd(
    num_blocks: int,
    dim: int,
    rank: int = 4,
    seed: int = 0,
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Build a flat safetensors-style state dict targeting attn.weight."""
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    for b in range(num_blocks):
        base = f"{prefix}transformer_blocks.{b}.attn"
        sd[f"{base}.lora_A.weight"] = torch.randn(
            rank,
            dim,
            generator=g,
            dtype=torch.float32,
        )
        sd[f"{base}.lora_B.weight"] = torch.randn(
            dim,
            rank,
            generator=g,
            dtype=torch.float32,
        )
    return sd


def _make_lora(
    num_blocks: int,
    dim: int,
    rank: int = 4,
    seed: int = 0,
    prefix: str = "",
) -> LoRA:
    """Build a LoRA targeting attn.weight across all blocks."""
    sd = _make_lora_sd(num_blocks, dim, rank=rank, seed=seed, prefix=prefix)
    return LoRA.from_state_dict(state_dict=sd)


def _request_loras(
    strategy: ModelOffloader,
    loras: Sequence[tuple[LoRA, float]],
    *,
    mode: LoRAMode = "merge",
) -> None:
    normalized = strategy._normalize_loras(
        [lora for lora, _strength in loras],
        lora_strengths=[strength for _lora, strength in loras],
    )
    _LORA_REQUESTS[strategy] = (normalized, mode)


_LORA_REQUESTS: dict[
    ModelOffloader,
    tuple[list[tuple[LoRA, float]], LoRAMode],
] = {}


def _activate(
    strategy: ModelOffloader,
    device: torch.device | str,
    **kwargs: object,
) -> None:
    loras, mode = _LORA_REQUESTS.pop(strategy, ([], "merge"))
    strategy.activate(
        device,
        loras=[lora for lora, _strength in loras],
        lora_strengths=[strength for _lora, strength in loras],
        lora_mode=mode,
        **kwargs,
    )


def _expected_merged_weight(
    base: torch.Tensor,
    loras: list[tuple[LoRA, float]],
    block_idx: int,
    qual: str,
) -> torch.Tensor:
    """Compute the target weight by summing all LoRA deltas onto the base."""
    out = base.clone()
    target_name = f"transformer_blocks.{block_idx}.{qual}"
    for lora, strength in loras:
        factors = lora.targets.get(target_name)
        if factors is None:
            continue
        a, b = _factor_tensors(factors)
        out.addmm_(
            b.to(device=out.device, dtype=out.dtype),
            a.to(device=out.device, dtype=out.dtype),
            alpha=strength,
        )
    return out


def _expected_routed_output(
    model: nn.Module,
    x: torch.Tensor,
    loras: list[tuple[LoRA, float]],
) -> torch.Tensor:
    """Manual routed baseline using F.linear to bypass installed hooks."""
    h = F.linear(x, model.embed.weight.to(x.device))
    for i, blk in enumerate(model.transformer_blocks):
        base_attn = F.linear(h, blk.attn.weight.to(h.device))
        target_name = f"transformer_blocks.{i}.attn.weight"
        a_parts = []
        b_parts = []
        for lora, strength in loras:
            factors = lora.targets.get(target_name)
            if factors is None:
                continue
            a, b = _factor_tensors(factors)
            a_parts.append(a.to(device=h.device, dtype=h.dtype))
            b_part = b.to(device=h.device, dtype=h.dtype).clone()
            b_part.mul_(strength)
            b_parts.append(b_part)
        if a_parts:
            a_fused = torch.cat(a_parts, dim=0)
            b_fused = torch.cat(b_parts, dim=1)
            base_attn = base_attn + (h @ a_fused.T) @ b_fused.T
        h = F.linear(base_attn, blk.ff.weight.to(h.device))
    return F.linear(h, model.head.weight.to(h.device))


def _make_strategy(model: nn.Module) -> ModelOffloader:
    """Shorthand for constructing the strategy.

    The block-streaming residency policy is now supplied at activation;
    pair this with :func:`_strategy_stream_config` on the matching
    ``activate``/``use`` call.
    """
    return _make_model_offloader(model, blocks_attr=["transformer_blocks"])


def _strategy_stream_config(
    model: nn.Module,
    num_resident_blocks: int | None = None,
) -> StreamConfig:
    """Residency policy matching :func:`_make_strategy`'s old defaults.

    Defaults to all-but-one blocks resident (the old ``blocks_to_swap=1``
    shape) so streaming is engaged regardless of the model's depth."""
    if num_resident_blocks is None:
        num_resident_blocks = len(model.transformer_blocks) - 1
    return StreamConfig(num_resident_blocks=num_resident_blocks)


def _has_post_copy_hook(strategy: ModelOffloader, target_key: str) -> bool:
    """Check whether a merge hook is installed for the given target."""
    if target_key not in strategy.param_names:
        return False
    param_name = target_key
    component = strategy._composite.component_for_param_name(param_name)
    if isinstance(component, PinnedComponent):
        instance = component._instance
        return instance.post_copy_hook_key(param_name) in instance._post_copy_hooks
    if isinstance(component, StreamedComponent):
        instance, local = component._resolve_param_name(param_name)
        return instance.post_copy_hook_key(local) in instance._post_copy_hooks
    return False


def _activate_loras_for_test(
    strategy: ModelOffloader,
) -> int:
    loras, mode = _LORA_REQUESTS.pop(strategy, ([], "merge"))
    if mode == "merge":
        targets = strategy._group_lora_factors_by_param_name(loras)
        try:
            strategy._register_merge_lora_hooks(torch.device("cuda"), targets)
        except BaseException:
            strategy._clear_active_loras()
            raise
        return len(targets)
    before = len(strategy._lora_hook_handles)
    try:
        strategy._register_routed_lora_hooks(loras, torch.device("cpu"))
        return len(strategy._lora_hook_handles) - before
    finally:
        strategy._clear_active_lora_hooks()
        strategy._deactivate_loras()


# ---------------------------------------------------------------------------
# LoRA construction validation
# ---------------------------------------------------------------------------


class TestLoRAConstruction:
    def test_unpaired_a_factor(self) -> None:
        sd = {"transformer_blocks.0.attn.lora_A.weight": torch.randn(4, 16)}
        with pytest.raises(ValueError, match="Unpaired"):
            LoRA.from_state_dict(state_dict=sd)

    def test_unpaired_b_factor(self) -> None:
        sd = {"transformer_blocks.0.attn.lora_B.weight": torch.randn(16, 4)}
        with pytest.raises(ValueError, match="Unpaired"):
            LoRA.from_state_dict(state_dict=sd)

    def test_rejects_non_floating_factor_dtype(self) -> None:
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.zeros(4, 16, dtype=torch.int32),
            "transformer_blocks.0.attn.lora_B.weight": torch.zeros(16, 4, dtype=torch.int32),
        }
        with pytest.raises(ValueError, match="floating-point"):
            LoRA.from_state_dict(state_dict=sd)

    def test_rejects_rank_mismatch(self) -> None:
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.attn.lora_B.weight": torch.randn(16, 8),
        }
        with pytest.raises(ValueError, match="shape mismatch"):
            LoRA.from_state_dict(state_dict=sd)

    def test_rejects_non_2d_factor(self) -> None:
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.randn(4),
            "transformer_blocks.0.attn.lora_B.weight": torch.randn(16, 4),
        }
        with pytest.raises(ValueError, match="shape mismatch"):
            LoRA.from_state_dict(state_dict=sd)

    def test_factors_are_pinned(self) -> None:
        lora = _make_lora(4, 16)
        for factor in lora.targets.values():
            a, b = _factor_tensors(factor)
            assert a.is_pinned()
            assert b.is_pinned()

    def test_factor_pinned_params_build_instance_without_repin(self) -> None:
        """Streaming keystone: a factor's pinned params drop straight into a
        PinnedModuleInstance — the same PinnedParam objects, no re-pin."""
        lora = _make_lora(1, 16, rank=4)
        factor = lora.targets["transformer_blocks.0.attn.weight"]
        assert isinstance(factor.a, PinnedParam)
        assert isinstance(factor.b, PinnedParam)

        holder = nn.Module()
        holder.register_parameter("a", factor.a.make_cpu_param())
        holder.register_parameter("b", factor.b.make_cpu_param())
        instance = PinnedModuleInstance(
            module=holder,
            params={"a": factor.a, "b": factor.b},
            buffers={},
        )
        # The streaming target pool will fill from these exact pinned
        # objects; nothing is cloned or re-pinned on the way to an instance.
        assert instance.params["a"] is factor.a
        assert instance.params["b"] is factor.b

    def test_cache_bytes(self) -> None:
        lora = _make_lora(4, 16, rank=4)
        expected = 4 * (4 * 16 + 16 * 4) * 4  # 4 blocks * 2 factors * float32
        assert lora.cache_bytes == expected

    def test_keys_used_verbatim(self) -> None:
        # Keys are used as-is — no built-in remapping. A prefixed key stays
        # prefixed; stripping it (e.g. ComfyUI's ``diffusion_model.``) is the
        # caller's job in the factory that produces the state dict.
        lora = _make_lora(1, 16, prefix="diffusion_model.")
        assert "diffusion_model.transformer_blocks.0.attn.weight" in lora.targets
        assert "transformer_blocks.0.attn.weight" not in lora.targets

    def test_from_state_dict_dtype_casts_factors_at_build(self) -> None:
        # Routed callers pass the model's compute dtype so factors stream and
        # reside at that width (and the hook's per-forward cast is a no-op).
        sd = _make_lora_sd(num_blocks=2, dim=16, seed=1)  # fp32 factors
        store = LoRA.from_state_dict(state_dict=sd, dtype=torch.bfloat16)
        for factor in store.targets.values():
            assert all(
                tensor.dtype == torch.bfloat16
                for tensor in _factor_tensors(factor)
            )
        # Default keeps the stored dtype.
        kept = LoRA.from_state_dict(state_dict=sd)
        for factor in kept.targets.values():
            assert all(
                tensor.dtype == torch.float32
                for tensor in _factor_tensors(factor)
            )

    def test_targets_are_cached_and_immutable(self) -> None:
        lora = _make_lora(1, 16)
        targets = lora.targets
        assert lora.targets is targets
        with pytest.raises(TypeError):
            targets["other.weight"] = next(iter(targets.values()))  # type: ignore[index]

    def test_from_state_dict_rejects_non_floating_dtype(self) -> None:
        # A non-floating dtype would slip past the float-factor validation
        # (which runs on the original tensors) and silently produce int factors.
        sd = _make_lora_sd(num_blocks=1, dim=16, seed=1)
        with pytest.raises(ValueError, match="floating-point"):
            LoRA.from_state_dict(state_dict=sd, dtype=torch.int8)

    def test_factor_holders_resolves_target_to_holder_modules(self) -> None:
        # The routed seam: a target weight key maps to its (lora_A, lora_B)
        # holder *modules* (stable identity), not the churning weight tensors.
        sd = _make_lora_sd(num_blocks=2, dim=16, seed=2)
        store = LoRA.from_state_dict(state_dict=sd)
        a, b = store.factor_holders("transformer_blocks.1.attn.weight")
        same_a, same_b = store.factor_holders(
            "transformer_blocks.1.attn.weight"
        )
        assert a is same_a
        assert b is same_b
        assert a.get_parameter("weight").shape == (4, 16)
        assert b.get_parameter("weight").shape == (16, 4)


# ---------------------------------------------------------------------------
# LoRA request validation
# ---------------------------------------------------------------------------


class TestActivationLoraValidation:
    def test_lora_strengths_default_to_one(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16)
        assert s._normalize_loras([lora]) == [(lora, 1.0)]

    def test_accepts_lora_strengths(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16)
        assert s._normalize_loras([lora], lora_strengths=[0.25]) == [
            (lora, 0.25),
        ]

    def test_rejects_lora_strength_length_mismatch(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        with pytest.raises(ValueError, match="same length"):
            s._normalize_loras(
                [_make_lora(4, 16)],
                lora_strengths=[],
            )

    def test_rejects_tuple_pairs(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        with pytest.raises(TypeError, match="LoRA instances"):
            s._normalize_loras(  # type: ignore[list-item]
                [(_make_lora(4, 16), 1.0)],
            )

    def test_rejects_duplicate_lora_instance(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16)
        with pytest.raises(ValueError, match="same LoRA instance"):
            s._normalize_loras([lora, lora])

    def test_invalid_lora_mode_releases_activation_claim(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        with pytest.raises(ValueError, match="lora_mode"):
            s.activate("cpu", lora_mode="invalid")  # type: ignore[arg-type]

        with activated_model(s, "cpu") as active:
            assert active is m

    def test_target_shape_mismatch_is_deferred_until_apply(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.attn.lora_B.weight": torch.randn(8, 4),
        }
        _request_loras(s, [(LoRA.from_state_dict(state_dict=sd), 1.0)])
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")

    def test_accepts_fp32_lora_target(self) -> None:
        m = _make_bf16_model().to(torch.float32)
        for p in m.parameters():
            p.requires_grad = False
        s = _make_strategy(m)
        _request_loras(s, [(_make_lora(4, 16), 1.0)])
        assert not _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")

    def test_non_block_targets_matched(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16),
            "embed.lora_B.weight": torch.randn(16, 4),
        }
        _request_loras(s, [(LoRA.from_state_dict(state_dict=sd), 1.0)])
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "embed.weight")

    def test_non_block_tied_alias_target_matched(self) -> None:
        m = _make_tied_non_block_model(dtype=torch.bfloat16)
        s = _make_strategy(m)
        sd = {
            "head.lora_A.weight": torch.randn(4, 16),
            "head.lora_B.weight": torch.randn(16, 4),
        }
        _request_loras(s, [(LoRA.from_state_dict(state_dict=sd), 1.0)], mode="merge")
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "head.weight")
        assert _has_post_copy_hook(s, "embed.weight")

    def test_rejects_duplicate_tied_alias_targets(self) -> None:
        m = _make_tied_non_block_model(dtype=torch.bfloat16)
        s = _make_strategy(m)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16),
            "embed.lora_B.weight": torch.randn(16, 4),
            "head.lora_A.weight": torch.randn(4, 16),
            "head.lora_B.weight": torch.randn(16, 4),
        }
        _request_loras(s, [(LoRA.from_state_dict(state_dict=sd), 1.0)], mode="merge")
        with pytest.raises(RuntimeError, match="shared LoRA targets"):
            _activate_loras_for_test(s)
        assert not _has_post_copy_hook(s, "embed.weight")
        assert not _has_post_copy_hook(s, "head.weight")

    def test_streamed_block_shared_submodule_alias_target_matched(self) -> None:
        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Linear(16, 16, bias=False)
                self.b = self.a

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.b(self.a(x))

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer_blocks = nn.ModuleList([Block(), Block()])

        m = M().to(torch.bfloat16)
        m.requires_grad_(False)
        s = _make_strategy(m)
        sd = {
            "transformer_blocks.0.b.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.b.lora_B.weight": torch.randn(16, 4),
        }
        _request_loras(s, [(LoRA.from_state_dict(state_dict=sd), 1.0)], mode="merge")
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "transformer_blocks.0.b.weight")
        assert _has_post_copy_hook(s, "transformer_blocks.0.a.weight")

    def test_exact_keys_match(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16)
        _request_loras(s, [(lora, 1.0)])
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")

    def test_prefixed_keys_rejected(self) -> None:
        # Keys are verbatim, so a ``diffusion_model.``-prefixed adapter does not
        # match the model's params — the caller must strip it before building.
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16, prefix="diffusion_model.")
        _request_loras(s, [(lora, 1.0)])
        with pytest.raises(ValueError, match="LoRA target .* is not managed"):
            _activate_loras_for_test(s)
        assert not _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")

    def test_target_keys_are_not_canonicalized(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {
            "transformer_blocks.0.attn.base_layer.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.attn.base_layer.lora_B.weight": torch.randn(16, 4),
        }
        _request_loras(s, [(LoRA.from_state_dict(state_dict=sd), 1.0)])

        with pytest.raises(ValueError, match="LoRA target .* is not managed"):
            _activate_loras_for_test(s)
        assert not _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")
        assert not _has_post_copy_hook(
            s,
            "transformer_blocks.0.attn.base_layer.weight",
        )

    def test_merge_mode_activation_rejects_cpu(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16)
        _request_loras(s, [(lora, 1.0)], mode="merge")
        with pytest.raises(ValueError, match="merge mode requires CUDA"):
            _activate(s, "cpu", stream_config=_strategy_stream_config(m))
        _assert_lora_routing_available(lora)

    def test_clear_active_loras_clears_previous_merge_hooks(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        _request_loras(s, [(_make_lora(4, 16, rank=4), 1.0)])
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")
        s._clear_active_loras()
        assert not _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")

    def test_merge_hooks_can_share_lora_with_routed_runtime(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16)
        _request_loras(s, [(lora, 1.0)])
        _activate_loras_for_test(s)
        try:
            assert _has_post_copy_hook(
                s,
                "transformer_blocks.0.attn.weight",
            )
            lora.activate("cpu", schedule_model=m)
            lora.deactivate()
        finally:
            s._clear_active_lora_hooks()

        _assert_lora_routing_available(lora)

    def test_accepts_quanto_target_in_merge_mode(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        m = _make_bf16_model()
        rows = cols = 16
        data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16)
        qt = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            data,
            scale,
            None,
        )
        m.embed.weight = nn.Parameter(qt, requires_grad=False)

        s = _make_strategy(m)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16),
            "embed.lora_B.weight": torch.randn(16, 4),
        }
        _request_loras(s, [(LoRA.from_state_dict(state_dict=sd), 1.0)], mode="merge")
        assert _activate_loras_for_test(s) == 1
        assert _has_post_copy_hook(s, "embed.weight")

        # routed mode must still accept it.
        _request_loras(s, [(LoRA.from_state_dict(state_dict=sd), 1.0)], mode="routed")
        route_count = _activate_loras_for_test(s)
        assert route_count == 1

    def test_merge_mode_defers_regular_non_addmm_dtype_until_apply(self) -> None:
        m = _make_bf16_model()
        m.embed.weight = nn.Parameter(
            torch.zeros(16, 16, dtype=torch.int32),
            requires_grad=False,
        )
        s = _make_strategy(m)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16),
            "embed.lora_B.weight": torch.randn(16, 4),
        }
        _request_loras(s, [(LoRA.from_state_dict(state_dict=sd), 1.0)], mode="merge")
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "embed.weight")

    def test_accepts_fp16_base(self) -> None:
        m = _make_bf16_model().to(torch.float16)
        for p in m.parameters():
            p.requires_grad = False
        s = _make_strategy(m)
        assert "embed.weight" in s.param_names

    def test_routed_mode_cpu_activation_uses_hooks(self) -> None:
        m = _make_bf16_model(num_blocks=2).to(torch.float32)
        for p in m.parameters():
            p.requires_grad = False
        loras = [(_make_lora(2, 16, seed=9), 0.75)]
        s = _make_strategy(m)
        _request_loras(s, loras, mode="routed")

        x = torch.randn(2, 16)
        _activate(s, "cpu", stream_config=_strategy_stream_config(m))
        try:
            actual = m(x)
            expected = _expected_routed_output(m, x, loras)
            assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
            assert not _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")
        finally:
            s.deactivate()

        assert s._active_loras == []


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @CUDA
    def test_activate_runs_components(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        _request_loras(s, [(_make_lora(4, 16), 1.0)])
        try:
            _activate(s, "cuda", stream_config=_strategy_stream_config(m))
            assert m.embed.weight.is_cuda
            assert m.head.weight.is_cuda
        finally:
            s.deactivate()

    @CUDA
    def test_deactivate_returns_to_pinned(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        _request_loras(s, [(_make_lora(4, 16), 1.0)])
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))
        s.deactivate()
        assert m.embed.weight.is_pinned()
        assert m.head.weight.is_pinned()

    @CUDA
    def test_reactivation_with_different_loras(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        _request_loras(s, [(_make_lora(4, 16, seed=1), 1.0)])
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))
        s.deactivate()
        _request_loras(s, [(_make_lora(4, 16, seed=2), 1.0)])
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))
        s.deactivate()
        assert m.embed.weight.is_pinned()

    @CUDA
    def test_base_only_reactivation_does_not_reuse_previous_merge_hooks(self) -> None:
        m = _make_bf16_model(num_blocks=4, dim=16)
        base_embed = m.embed.weight.detach().clone()
        base_block = m.transformer_blocks[0].attn.weight.detach().clone()

        sd = _make_lora_sd(num_blocks=4, dim=16, seed=3)
        g = torch.Generator().manual_seed(303)
        sd["embed.lora_A.weight"] = torch.randn(
            4,
            16,
            generator=g,
            dtype=torch.float32,
        )
        sd["embed.lora_B.weight"] = torch.randn(
            16,
            4,
            generator=g,
            dtype=torch.float32,
        )
        s = _make_strategy(m)
        _request_loras(s, [(LoRA.from_state_dict(state_dict=sd), 1.0)], mode="merge")
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))
        s.deactivate()

        _request_loras(s, [])
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))
        try:
            torch.cuda.synchronize()
            torch.testing.assert_close(
                m.embed.weight.detach().cpu(),
                base_embed,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                m.transformer_blocks[0].attn.weight.detach().cpu(),
                base_block,
                rtol=0.0,
                atol=0.0,
            )
        finally:
            s.deactivate()

    @CUDA
    def test_activate_with_no_loras_runs_base_only(self) -> None:
        m = _make_bf16_model()
        captured = m.transformer_blocks[0].attn.weight.detach().clone()
        s = _make_strategy(m)
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            for blk in m.transformer_blocks:
                x = blk(x)
            torch.cuda.synchronize()
            actual = m.transformer_blocks[0].attn.weight.detach()
            assert torch.allclose(
                actual.cpu(),
                captured,
                rtol=0.0,
                atol=0.0,
            ), "no LoRAs must leave base weights unmodified"
        finally:
            s.deactivate()


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


class TestMergeCorrectness:
    @CUDA
    def test_merged_weights_match_manual_baseline(self) -> None:
        m = _make_bf16_model(num_blocks=4, dim=16)
        captured_base = {i: m.transformer_blocks[i].attn.weight.detach().clone() for i in range(4)}

        loras = [
            (_make_lora(num_blocks=4, dim=16, seed=10), 0.5),
            (_make_lora(num_blocks=4, dim=16, seed=20), 0.25),
        ]
        s = _make_strategy(m)
        _request_loras(s, loras)
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            for blk in m.transformer_blocks:
                x = blk(x)
            torch.cuda.synchronize()
            for i in range(4):
                _ = m.transformer_blocks[i](torch.randn(2, 16, dtype=torch.bfloat16, device="cuda"))
                torch.cuda.synchronize()
                expected = _expected_merged_weight(
                    captured_base[i].to("cuda"),
                    loras,
                    i,
                    "attn.weight",
                )
                actual = m.transformer_blocks[i].attn.weight.detach()
                assert torch.allclose(actual, expected, rtol=0.01, atol=0.01), (
                    f"block {i} merged weight mismatch:\n"
                    f"  expected: {expected.flatten()[:4]}\n"
                    f"  actual:   {actual.flatten()[:4]}"
                )
        finally:
            s.deactivate()

    @CUDA
    def test_non_block_lora_merges_correctly(self) -> None:
        """LoRA targeting embed (non-block) should be merged at activate."""
        m = _make_bf16_model(num_blocks=4, dim=16)
        captured_embed = m.embed.weight.detach().clone()

        g = torch.Generator().manual_seed(99)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16, generator=g, dtype=torch.float32),
            "embed.lora_B.weight": torch.randn(16, 4, generator=g, dtype=torch.float32),
        }
        lora = LoRA.from_state_dict(state_dict=sd)
        s = _make_strategy(m)
        _request_loras(s, [(lora, 0.5)])
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))
        try:
            factor = lora.targets["embed.weight"]
            a, b = _factor_tensors(factor)
            expected = (captured_embed + 0.5 * (b.to(torch.bfloat16) @ a.to(torch.bfloat16))).to("cuda")
            actual = m.embed.weight.detach()
            assert torch.allclose(actual, expected, rtol=0.01, atol=0.01), (
                f"non-block merge mismatch:\n  expected: {expected.flatten()[:4]}\n  actual:   {actual.flatten()[:4]}"
            )
        finally:
            s.deactivate()

    @CUDA
    def test_base_layer_named_target_merges_with_exact_keys(self) -> None:
        """A model whose weight lives at a nested ``.base_layer.`` path (e.g.
        PEFT-wrapped) merges when the LoRA keys that exact path. The offloader
        does no key remapping — the caller builds keys that match the model."""

        class PEFTBlock(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.attn = nn.Module()
                self.attn.base_layer = nn.Linear(dim, dim, bias=False)
                self.ff = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.ff(self.attn.base_layer(x))

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer_blocks = nn.ModuleList([PEFTBlock(16) for _ in range(4)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for blk in self.transformer_blocks:
                    x = blk(x)
                return x

        m = M().to(torch.bfloat16)
        m.requires_grad_(False)

        captured_base = {i: m.transformer_blocks[i].attn.base_layer.weight.detach().clone() for i in range(4)}

        # Keys match the model's real paths, ``.base_layer.`` and all.
        g = torch.Generator().manual_seed(42)
        sd: dict[str, torch.Tensor] = {}
        for b in range(4):
            base = f"transformer_blocks.{b}.attn.base_layer"
            sd[f"{base}.lora_A.weight"] = torch.randn(4, 16, generator=g)
            sd[f"{base}.lora_B.weight"] = torch.randn(16, 4, generator=g)
        lora = LoRA.from_state_dict(state_dict=sd)

        s = _make_strategy(m)
        _request_loras(s, [(lora, 0.7)])
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            m(x)
            torch.cuda.synchronize()
            for i in range(4):
                _ = m.transformer_blocks[i](torch.randn(2, 16, dtype=torch.bfloat16, device="cuda"))
                torch.cuda.synchronize()
                expected = _expected_merged_weight(
                    captured_base[i].to("cuda"),
                    [(lora, 0.7)],
                    i,
                    "attn.base_layer.weight",
                )
                actual = m.transformer_blocks[i].attn.base_layer.weight.detach()
                assert torch.allclose(actual, expected, rtol=0.01, atol=0.01), f"block {i} base_layer merge mismatch"
        finally:
            s.deactivate()


class TestLoRATransform:
    def test_validate_target_accepts_dense_without_mutation(self) -> None:
        param = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        before = param.detach().clone()
        a = torch.randn(2, 8)
        b = torch.randn(4, 2)
        transform = LoRATransform([ScaledLoRAFactor(a, b, 0.5)])

        transform.validate_target(param)

        torch.testing.assert_close(param, before)

    def test_validate_target_rejects_shape_mismatch(self) -> None:
        param = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        a = torch.randn(2, 8)
        b = torch.randn(3, 2)
        transform = LoRATransform([ScaledLoRAFactor(a, b, 0.5)])

        with pytest.raises(ValueError, match="B@A produces"):
            transform.validate_target(param)

    def test_dense_transform_mutates_param_in_place(self) -> None:
        param = nn.Parameter(torch.randn(4, 8), requires_grad=False)
        before = param.detach().clone()
        a = torch.randn(2, 8)
        b = torch.randn(4, 2)
        transform = LoRATransform([ScaledLoRAFactor(a, b, 0.5)])

        transform.apply(param)

        expected = before.clone()
        expected.addmm_(b, a, alpha=0.5)
        torch.testing.assert_close(param, expected)

    def test_regular_non_addmm_dtype_raises_on_apply(self) -> None:
        param = nn.Parameter(torch.zeros(4, 8, dtype=torch.int32), requires_grad=False)
        a = torch.randn(2, 8)
        b = torch.randn(4, 2)
        transform = LoRATransform([ScaledLoRAFactor(a, b, 0.5)])

        with pytest.raises(ValueError, match="dense in-place addmm requires"):
            transform.apply(param)

    def test_quanto_transform_requantizes_param_in_place(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols, rank = 4, 8, 2
        data = torch.randint(-32, 32, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25)
        qt = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            data,
            scale,
            None,
        )
        param = nn.Parameter(qt, requires_grad=False)
        a = torch.randn(rank, cols)
        b = torch.randn(rows, rank)
        transform = LoRATransform([ScaledLoRAFactor(a, b, 0.5)])
        original_param = param
        original_packed_ptr = param.data._data.data_ptr()
        expected_dense = qt.dequantize().to(torch.float32)

        transform.apply(param)

        expected_dense.addmm_(b.to(torch.float32), a.to(torch.float32), alpha=0.5)
        expected_packed = (expected_dense / scale.to(torch.float32)).round().clamp(-128, 127).to(torch.int8)
        assert param is original_param
        assert param.data._data.data_ptr() == original_packed_ptr
        assert isinstance(param.data, WeightQBytesTensor)
        torch.testing.assert_close(param.data._data, expected_packed)
        torch.testing.assert_close(param.data._scale, scale)

    @CUDA
    def test_non_block_quanto_merge_requantizes_on_activate(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        m = _make_bf16_model(num_blocks=1, dim=16)
        rows = cols = 16
        rank = 4
        data = torch.randint(-32, 32, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25)
        qt = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            data,
            scale,
            None,
        )
        m.embed.weight = nn.Parameter(qt, requires_grad=False)
        sd = {
            "embed.lora_A.weight": torch.randn(rank, cols),
            "embed.lora_B.weight": torch.randn(rows, rank),
        }
        lora = LoRA.from_state_dict(state_dict=sd)
        factor = lora.targets["embed.weight"]
        a, b = _factor_tensors(factor)
        # Compute the reference on CUDA, matching the device the offloader
        # merges on. A CPU reference flips occasional int8 elements at
        # quantization bucket edges (CPU vs CUDA round-to-nearest), and the
        # comparison is exact — so the device must match to be deterministic.
        qt_cuda = qt.cuda()
        expected_dense = qt_cuda.dequantize().to(torch.float32)
        expected_dense.addmm_(b.cuda().to(torch.float32), a.cuda().to(torch.float32), alpha=0.5)
        expected_packed = (expected_dense / scale.cuda().to(torch.float32)).round().clamp(-128, 127).to(torch.int8)

        s = _make_strategy(m)
        _request_loras(s, [(lora, 0.5)], mode="merge")
        _activate(s,
            "cuda",
            stream_config=_strategy_stream_config(m, num_resident_blocks=1),
        )
        try:
            merged_qt = m.embed.weight.data
            assert isinstance(merged_qt, WeightQBytesTensor)
            torch.testing.assert_close(merged_qt._data, expected_packed)
            torch.testing.assert_close(merged_qt._scale.cpu(), scale)
        finally:
            s.deactivate()

    @CUDA
    def test_streamed_quanto_merge_requantizes_pool_param_in_place(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        m = _make_bf16_model(num_blocks=2, dim=16)
        rows = cols = 16
        rank = 4
        scales: list[torch.Tensor] = []
        original_qt: WeightQBytesTensor | None = None
        for block in m.transformer_blocks:
            data = torch.randint(-32, 32, (rows, cols), dtype=torch.int8)
            scale = torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25)
            qt = WeightQBytesTensor.create(
                quanto.qint8,
                0,
                (rows, cols),
                (cols, 1),
                data,
                scale,
                None,
            )
            if original_qt is None:
                original_qt = qt
            scales.append(scale)
            block.attn.weight = nn.Parameter(qt, requires_grad=False)
        assert original_qt is not None

        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.randn(rank, cols),
            "transformer_blocks.0.attn.lora_B.weight": torch.randn(rows, rank),
        }
        lora = LoRA.from_state_dict(state_dict=sd)
        factor = lora.targets["transformer_blocks.0.attn.weight"]
        a, b = _factor_tensors(factor)
        expected_dense = original_qt.dequantize().to(torch.float32)
        expected_dense.addmm_(b.to(torch.float32), a.to(torch.float32), alpha=0.5)
        expected_packed = (expected_dense / scales[0].to(torch.float32)).round().clamp(-128, 127).to(torch.int8)

        s = _make_strategy(m)
        _request_loras(s, [(lora, 0.5)], mode="merge")
        _activate(s,
            "cuda",
            stream_config=_strategy_stream_config(m, num_resident_blocks=1),
        )
        try:
            streamer = streamed_components(s)[0]
            streamer._load_block(0)
            merged_qt = m.transformer_blocks[0].attn.weight.data
            assert isinstance(merged_qt, WeightQBytesTensor)
            torch.testing.assert_close(merged_qt._data.cpu(), expected_packed)
            torch.testing.assert_close(merged_qt._scale.cpu(), scales[0])
        finally:
            s.deactivate()


class TestPermanentMerge:
    def test_can_share_lora_with_an_active_routed_use(self) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16).to(torch.float32)
        lora = _make_lora(num_blocks=2, dim=16)
        before = m.transformer_blocks[0].attn.weight.detach().clone()
        expected = _expected_merged_weight(
            before,
            [(lora, 1.0)],
            0,
            "attn.weight",
        )

        lora.activate(
            "cpu",
            schedule_model=_make_bf16_model(num_blocks=2, dim=16),
        )
        try:
            assert merge_lora(m, [(lora, 1.0)]) == 2
        finally:
            lora.deactivate()

        torch.testing.assert_close(m.transformer_blocks[0].attn.weight, expected)

    def test_merge_leaves_routed_runtime_available(self) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16).to(torch.float32)
        lora = _make_lora(num_blocks=2, dim=16)

        assert merge_lora(m, [(lora, 1.0)]) == 2
        _assert_lora_routing_available(lora)

    def test_rejects_unknown_targets_without_mutation(self) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16).to(torch.float32)
        before = {
            name: param.detach().clone()
            for name, param in m.named_parameters()
        }
        lora = _make_lora(
            num_blocks=2,
            dim=16,
            prefix="missing.",
        )

        with pytest.raises(ValueError, match="not parameters in the model"):
            merge_lora(m, [(lora, 1.0)])

        for name, param in m.named_parameters():
            torch.testing.assert_close(param, before[name])
        _assert_lora_routing_available(lora)

    def test_multiple_loras_on_one_target_count_one_parameter(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = nn.Linear(3, 3, bias=False)

        m = M()
        m.requires_grad_(False)
        base = m.target.weight.detach().clone()

        def make_lora(seed: int) -> LoRA:
            g = torch.Generator().manual_seed(seed)
            return LoRA.from_state_dict(
                {
                    "target.lora_A.weight": torch.randn(1, 3, generator=g),
                    "target.lora_B.weight": torch.randn(3, 1, generator=g),
                }
            )

        first = make_lora(1)
        second = make_lora(2)
        expected = base.clone()
        for lora in (first, second):
            a, b = _factor_tensors(lora.targets["target.weight"])
            expected.addmm_(b, a)

        assert merge_lora(m, [(first, 1.0), (second, 1.0)]) == 1
        torch.testing.assert_close(m.target.weight, expected)

    def test_shape_preflight_prevents_partial_merge(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.first = nn.Linear(3, 3, bias=False)
                self.second = nn.Linear(3, 3, bias=False)

        m = M()
        m.requires_grad_(False)
        first_before = m.first.weight.detach().clone()
        second_before = m.second.weight.detach().clone()
        lora = LoRA.from_state_dict(
            {
                "first.lora_A.weight": torch.randn(1, 3),
                "first.lora_B.weight": torch.randn(3, 1),
                "second.lora_A.weight": torch.randn(1, 3),
                "second.lora_B.weight": torch.randn(2, 1),
            }
        )

        with pytest.raises(ValueError, match="second.weight.*shape mismatch"):
            merge_lora(m, [(lora, 1.0)])

        torch.testing.assert_close(m.first.weight, first_before)
        torch.testing.assert_close(m.second.weight, second_before)

    def test_quanto_target_uses_dequant_requant_strategy(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        class M(nn.Module):
            def __init__(self, weight: torch.Tensor) -> None:
                super().__init__()
                self.target = nn.Linear(8, 4, bias=False)
                self.target.weight = nn.Parameter(weight, requires_grad=False)

        rows, cols, rank = 4, 8, 2
        data = torch.randint(-32, 32, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16).add_(0.25)
        qt = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (rows, cols),
            (cols, 1),
            data,
            scale,
            None,
        )
        m = M(qt)
        original_param = m.target.weight
        original_packed_ptr = original_param.data._data.data_ptr()
        sd = {
            "target.lora_A.weight": torch.randn(rank, cols),
            "target.lora_B.weight": torch.randn(rows, rank),
        }
        lora = LoRA.from_state_dict(state_dict=sd)
        factor = lora.targets["target.weight"]
        a, b = _factor_tensors(factor)

        expected_dense = qt.dequantize().to(torch.float32)
        expected_dense.addmm_(b.to(torch.float32), a.to(torch.float32), alpha=0.5)
        expected_packed = (expected_dense / scale.to(torch.float32)).round().clamp(-128, 127).to(torch.int8)

        merged = merge_lora(m, [(lora, 0.5)])

        assert merged == 1
        assert m.target.weight is original_param
        merged_qt = m.target.weight.data
        assert isinstance(merged_qt, WeightQBytesTensor)
        assert merged_qt._data.data_ptr() == original_packed_ptr
        assert merged_qt.qtype is quanto.qint8
        assert merged_qt.axis == 0
        assert tuple(merged_qt.size()) == (rows, cols)
        torch.testing.assert_close(merged_qt._data, expected_packed)
        torch.testing.assert_close(merged_qt._scale, scale)

    def test_multiple_loras_requantize_shared_target_once(self) -> None:
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        class M(nn.Module):
            def __init__(self, weight: torch.Tensor) -> None:
                super().__init__()
                self.target = nn.Linear(2, 2, bias=False)
                self.target.weight = nn.Parameter(weight, requires_grad=False)

        data = torch.zeros((2, 2), dtype=torch.int8)
        scale = torch.ones((2, 1), dtype=torch.float32)
        qt = WeightQBytesTensor.create(
            quanto.qint8,
            0,
            (2, 2),
            (2, 1),
            data,
            scale,
            None,
        )

        def make_lora() -> LoRA:
            return LoRA.from_state_dict(
                {
                    "target.lora_A.weight": torch.ones(1, 2),
                    "target.lora_B.weight": torch.full((2, 1), 0.6),
                }
            )

        m = M(qt)
        first = make_lora()
        second = make_lora()

        assert merge_lora(m, [(first, 1.0), (second, 1.0)]) == 1
        # Both 0.6 deltas are accumulated in dense space before the one
        # unit-scale int8 requantization: round(0.6 + 0.6) == 1. Requantizing
        # after each contribution instead would incorrectly produce 2.
        torch.testing.assert_close(
            m.target.weight.data.dequantize(),
            torch.ones(2, 2),
        )

    def test_tied_alias_target_merges_shared_storage(self) -> None:
        m = _make_tied_non_block_model(dtype=torch.float32)
        base = m.embed.weight.detach().clone()
        sd = {
            "head.lora_A.weight": torch.randn(4, 16),
            "head.lora_B.weight": torch.randn(16, 4),
        }
        lora = LoRA.from_state_dict(state_dict=sd)
        factor = lora.targets["head.weight"]
        a, b = _factor_tensors(factor)

        merged = merge_lora(m, [(lora, 0.25)])

        expected = base.clone()
        expected.addmm_(
            b.to(dtype=expected.dtype),
            a.to(dtype=expected.dtype),
            alpha=0.25,
        )
        assert merged == 1
        torch.testing.assert_close(m.embed.weight, expected)
        torch.testing.assert_close(m.head.weight, expected)
        assert m.head.weight is m.embed.weight

    def test_duplicate_tied_alias_targets_raise_before_mutation(self) -> None:
        m = _make_tied_non_block_model(dtype=torch.float32)
        before = m.embed.weight.detach().clone()
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16),
            "embed.lora_B.weight": torch.randn(16, 4),
            "head.lora_A.weight": torch.randn(4, 16),
            "head.lora_B.weight": torch.randn(16, 4),
        }

        with pytest.raises(ValueError, match="same tied parameter backing"):
            merge_lora(m, [(LoRA.from_state_dict(state_dict=sd), 1.0)])

        torch.testing.assert_close(m.embed.weight, before)
        assert m.head.weight is m.embed.weight

    def test_unmatched_unsupported_tensor_subclass_is_ignored(self) -> None:
        class UnknownTensor(torch.Tensor):
            pass

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = nn.Linear(16, 16, bias=False)
                self.other = nn.Linear(16, 16, bias=False)
                wrapped = torch.Tensor._make_subclass(
                    UnknownTensor,
                    torch.randn(16, 16),
                    False,
                )
                self.other.weight = nn.Parameter(wrapped, requires_grad=False)

        m = M()
        before = m.target.weight.detach().clone()
        sd = {
            "target.lora_A.weight": torch.randn(4, 16),
            "target.lora_B.weight": torch.randn(16, 4),
        }
        lora = LoRA.from_state_dict(state_dict=sd)
        factor = lora.targets["target.weight"]
        a, b = _factor_tensors(factor)

        merged = merge_lora(m, [(lora, 0.5)])

        expected = before.clone()
        expected.addmm_(b, a, alpha=0.5)
        assert merged == 1
        torch.testing.assert_close(m.target.weight, expected)


# ---------------------------------------------------------------------------
# Cleanup invariants
# ---------------------------------------------------------------------------


class TestRoutedMode:
    """Routed-mode LoRA: forward hook on the parent layer, base
    weight untouched. Math: y = base(x) + alpha * B * A * x.
    """

    def _expected_routed_output(
        self,
        model: nn.Module,
        x: torch.Tensor,
        loras: list[tuple[LoRA, float]],
    ) -> torch.Tensor:
        """Manual baseline: walk the block list using F.linear so we
        bypass any forward hooks installed on the layers (otherwise
        the expected calculation would also include the hook output
        and we'd be comparing the hook against itself)."""
        return _expected_routed_output(model, x, loras)

    def test_routed_accepts_linear_input_by_keyword(self) -> None:
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = nn.Linear(3, 3, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.target(input=x)

        m = M()
        m.requires_grad_(False)
        lora = LoRA.from_state_dict(
            {
                "target.lora_A.weight": torch.randn(1, 3),
                "target.lora_B.weight": torch.randn(3, 1),
            }
        )
        factor = lora.targets["target.weight"]
        a, b = _factor_tensors(factor)
        x = torch.randn(2, 3)
        strength = 0.5
        offloader = _make_model_offloader(m)

        offloader.activate(
            "cpu",
            loras=[lora],
            lora_strengths=[strength],
            lora_mode="routed",
        )
        try:
            actual = m(x)
            expected = F.linear(x, m.target.weight)
            expected += ((x @ a.T) * strength) @ b.T
            torch.testing.assert_close(actual, expected)
        finally:
            offloader.deactivate()

    @CUDA
    def test_routed_forward_matches_manual_baseline(self) -> None:
        # Routed mode: base weight stays exactly as constructed; the
        # LoRA contribution rides as a forward-hook addition.
        m = _make_bf16_model(num_blocks=3, dim=16)
        loras = [
            (_make_lora(num_blocks=3, dim=16, seed=11), 0.5),
            (_make_lora(num_blocks=3, dim=16, seed=22), 0.25),
        ]
        base_snapshots = [m.transformer_blocks[i].attn.weight.detach().clone() for i in range(3)]

        s = _make_strategy(m)
        _request_loras(s, loras, mode="routed")
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            actual = m(x)
            torch.cuda.synchronize()
            expected = self._expected_routed_output(m, x, loras)
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1), (
                f"routed forward mismatch:\n  expected: {expected.flatten()[:4]}\n  actual:   {actual.flatten()[:4]}"
            )
        finally:
            s.deactivate()

        # Base weights on CPU snapshots must equal the model's current
        # (post-deactivate) base weights — routed mode didn't mutate.
        for i in range(3):
            assert torch.equal(
                m.transformer_blocks[i].attn.weight.detach(),
                base_snapshots[i],
            ), f"routed mode mutated block {i} base weight"

    @CUDA
    def test_routed_clears_on_deactivate(self) -> None:
        # Hooks installed on activate must be removed on deactivate so
        # subsequent base-only forward sees the unaugmented model.
        m = _make_bf16_model(num_blocks=3, dim=16)
        s = _make_strategy(m)
        _request_loras(s, [(_make_lora(3, 16, seed=7), 1.0)], mode="routed")
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))
        x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
        with_lora = m(x).detach().clone()
        torch.cuda.synchronize()
        s.deactivate()

        # Re-activate without LoRAs; output should differ from with_lora
        # (the hooks should be gone).
        _request_loras(s, [], mode="routed")
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))
        try:
            base_only = m(x)
            torch.cuda.synchronize()
            assert not torch.allclose(with_lora, base_only, rtol=0.001, atol=0.001), (
                "deactivate did not remove routed hooks; base-only output still reflects LoRA contribution"
            )
        finally:
            s.deactivate()

    @CUDA
    def test_routed_mixed_ranks(self) -> None:
        # Multiple LoRAs targeting the same weight at different ranks
        # must produce the same forward output as the per-adapter route
        # math: each (A_i, B_i) is applied independently and summed,
        # output = base(x) + sum_i strength_i * (x @ A_i.T) @ B_i.T.
        m = _make_bf16_model(num_blocks=2, dim=16)
        # Two LoRAs targeting the same blocks with different ranks.
        lora_r4 = _make_lora(num_blocks=2, dim=16, rank=4, seed=101)
        lora_r8 = _make_lora(num_blocks=2, dim=16, rank=8, seed=202)
        loras = [(lora_r4, 0.6), (lora_r8, 0.3)]

        s = _make_strategy(m)
        _request_loras(s, loras, mode="routed")
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            actual = m(x)
            torch.cuda.synchronize()
            expected = self._expected_routed_output(m, x, loras)
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1), (
                f"mixed-rank routed output mismatch:\n"
                f"  expected: {expected.flatten()[:4]}\n"
                f"  actual:   {actual.flatten()[:4]}"
            )
        finally:
            s.deactivate()

    def test_routed_with_non_linear_target_raises(self) -> None:
        # Routed math assumes y = x @ W.T + bias. If a target's parent
        # is not nn.Linear, routed activation must reject so we don't
        # silently install a hook against an incompatible forward.
        class LinearLike(nn.Module):
            """nn.Linear-shaped weight but not an nn.Linear instance."""

            def __init__(self, dim: int) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(dim, dim, dtype=torch.bfloat16),
                    requires_grad=False,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.weight.T

        class Block(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.attn = LinearLike(dim)
                self.ff = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.ff(self.attn(x))

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer_blocks = nn.ModuleList([Block(16) for _ in range(2)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for blk in self.transformer_blocks:
                    x = blk(x)
                return x

        model = M().to(torch.bfloat16)
        for p in model.parameters():
            p.requires_grad = False

        s = _make_model_offloader(
            model,
            blocks_attr=["transformer_blocks"],
        )
        # Build a LoRA targeting attn.weight (LinearLike, not nn.Linear).
        lora = _make_lora(num_blocks=2, dim=16, seed=3)
        _request_loras(s, [(lora, 1.0)], mode="routed")
        with pytest.raises(ValueError, match=r"Routed LoRA mode requires nn\.Linear"):
            _activate_loras_for_test(s)
        # Merge mode should still work — it doesn't care about parent type.
        _request_loras(s, [(lora, 1.0)], mode="merge")
        _activate_loras_for_test(s)

    def test_routed_partial_failure_leaves_no_hooks(self) -> None:
        # Mid-loop route rejection (e.g., one non-Linear target after
        # some valid Linear targets) must NOT leave half-built active
        # hooks.
        class LinearLike(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn(dim, dim, dtype=torch.bfloat16),
                    requires_grad=False,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.weight.T

        class Block(nn.Module):
            def __init__(self, dim: int, normal_attn: bool) -> None:
                super().__init__()
                self.attn = nn.Linear(dim, dim, bias=False) if normal_attn else LinearLike(dim)
                self.ff = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.ff(self.attn(x))

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Block 0: normal Linear (passes routed validation).
                # Block 1: LinearLike (fails routed validation).
                self.transformer_blocks = nn.ModuleList([Block(16, normal_attn=True), Block(16, normal_attn=False)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for blk in self.transformer_blocks:
                    x = blk(x)
                return x

        model = M().to(torch.bfloat16)
        for p in model.parameters():
            p.requires_grad = False

        s = _make_model_offloader(
            model,
            blocks_attr=["transformer_blocks"],
        )
        lora = _make_lora(num_blocks=2, dim=16, seed=99)
        _request_loras(s, [(lora, 1.0)], mode="routed")
        with pytest.raises(ValueError, match=r"Routed LoRA mode requires nn\.Linear"):
            _activate_loras_for_test(s)

    @CUDA
    def test_routed_single_adapter(self) -> None:
        # Single-adapter case: one (A, B) pair, no summation. Forward
        # output must still match the manual baseline.
        m = _make_bf16_model(num_blocks=2, dim=16)
        loras = [(_make_lora(num_blocks=2, dim=16, seed=33), 0.7)]

        s = _make_strategy(m)
        _request_loras(s, loras, mode="routed")
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            actual = m(x)
            torch.cuda.synchronize()
            expected = self._expected_routed_output(m, x, loras)
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1)
        finally:
            s.deactivate()

    @pytest.mark.parametrize("target", ["embed", "head"])
    def test_routed_tied_weight_target_uses_exact_parent(self, target: str) -> None:
        # Standard tied embed/head pattern: one Parameter aliased at
        # multiple names. Routed mode is name-centric and does not
        # mutate the shared storage, so it should hook only the exact
        # parent module named by the LoRA target.
        model = _make_tied_non_block_model(dtype=torch.bfloat16)

        s = _make_model_offloader(
            model,
            blocks_attr=["transformer_blocks"],
        )
        # Build a LoRA that targets either alias of the tied weight.
        sd = {
            f"{target}.lora_A.weight": torch.randn(4, 16),
            f"{target}.lora_B.weight": torch.randn(16, 4),
        }
        lora = LoRA.from_state_dict(state_dict=sd)
        _request_loras(s, [(lora, 1.0)], mode="routed")
        active_loras, _mode = _LORA_REQUESTS.pop(s)
        s._register_routed_lora_hooks(active_loras, torch.device("cpu"))
        try:
            assert len(model.embed._forward_hooks) == (1 if target == "embed" else 0)
            assert len(model.head._forward_hooks) == (1 if target == "head" else 0)
        finally:
            s._clear_active_lora_hooks()
            s._deactivate_loras()

        # Merge mode also matches by name; it mutates the copied backing,
        # so the normal shared-storage effect is preserved.
        _request_loras(s, [(lora, 1.0)], mode="merge")
        _activate_loras_for_test(s)

    @CUDA
    def test_routed_with_bias_linear(self) -> None:
        # Routed math: hook adds alpha * B * A * x to the layer's output, which
        # already includes any bias from the base layer. Bias-having
        # Linears must produce the same output as a manual baseline
        # that goes through F.linear with the bias.
        class Block(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.attn = nn.Linear(dim, dim, bias=True)
                self.ff = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.ff(self.attn(x))

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(16, 16, bias=False)
                self.transformer_blocks = nn.ModuleList([Block(16) for _ in range(2)])
                self.head = nn.Linear(16, 16, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.embed(x)
                for blk in self.transformer_blocks:
                    h = blk(h)
                return self.head(h)

        m = M().to(torch.bfloat16)
        for p in m.parameters():
            p.requires_grad = False

        loras = [(_make_lora(num_blocks=2, dim=16, seed=55), 0.5)]
        s = _make_model_offloader(
            m,
            blocks_attr=["transformer_blocks"],
        )
        _request_loras(s, loras, mode="routed")
        _activate(s, "cuda")
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            actual = m(x)
            torch.cuda.synchronize()
            # Manual baseline: walk layers via F.linear (bypasses hooks)
            # with bias included on attn.
            h = F.linear(x, m.embed.weight.to(x.device))
            for i, blk in enumerate(m.transformer_blocks):
                base_attn = F.linear(
                    h,
                    blk.attn.weight.to(h.device),
                    blk.attn.bias.to(h.device),
                )
                lora = loras[0][0]
                strength = loras[0][1]
                factor = lora.targets[f"transformer_blocks.{i}.attn.weight"]
                a, b = _factor_tensors(factor)
                a_dev = a.to(device=h.device, dtype=h.dtype)
                b_dev = b.to(device=h.device, dtype=h.dtype)
                attn_out = base_attn + strength * (h @ a_dev.T @ b_dev.T)
                h = F.linear(attn_out, blk.ff.weight.to(h.device))
            expected = F.linear(h, m.head.weight.to(h.device))
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1)
        finally:
            s.deactivate()


class TestRoutedStreaming:
    """Routed LoRA on the streaming foundation (#31): per-LoRA stacked hooks
    read the LoRA's resident factors, co-scheduled with the base model."""

    def test_multi_lora_stacked_hooks_compose_additively(self) -> None:
        # One block (so the two adapters don't feed each other across blocks)
        # and a fully-linear tail, so LoRA residuals superpose:
        # y(L1+L2) == y(L1) + y(L2) - y0. Two LoRAs on the one attn.weight
        # install two stacked forward-POST hooks; additive hook chaining must
        # sum them with no central grouping.
        torch.manual_seed(0)
        m = _make_bf16_model(num_blocks=1, dim=16).to(torch.float32)
        x = torch.randn(3, 16)
        lora1 = _make_lora(num_blocks=1, dim=16, seed=1)
        lora2 = _make_lora(num_blocks=1, dim=16, seed=2)
        s = _make_model_offloader(m, blocks_attr=["transformer_blocks"])

        def routed_forward(
            loras: list[tuple[LoRA, float]],
        ) -> torch.Tensor:
            if loras:
                _request_loras(s, loras, mode="routed")
            _activate(s, torch.device("cpu"))
            try:
                return m(x).clone()
            finally:
                s.deactivate()

        y0 = routed_forward([])
        y1 = routed_forward([(lora1, 0.7)])
        y2 = routed_forward([(lora2, 1.3)])
        y12 = routed_forward([(lora1, 0.7), (lora2, 1.3)])
        torch.testing.assert_close(
            y12,
            y1 + y2 - y0,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_routed_nested_base_layer_path(self) -> None:
        # A weight nested under ``.base_layer.`` (e.g. PEFT-wrapped). Routed
        # resolves the hook onto that nested Linear and reads the factor from
        # the matching nested holder — both keyed at the exact model path.
        class PEFTBlock(nn.Module):
            def __init__(self, dim: int) -> None:
                super().__init__()
                self.attn = nn.Module()
                self.attn.base_layer = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.attn.base_layer(x)

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.transformer_blocks = nn.ModuleList([PEFTBlock(16) for _ in range(2)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for blk in self.transformer_blocks:
                    x = blk(x)
                return x

        m = M()
        for p in m.parameters():
            p.requires_grad = False
        base_w = [m.transformer_blocks[b].attn.base_layer.weight.detach().clone() for b in range(2)]
        # Keys match the model's real ``.base_layer.`` paths exactly.
        g = torch.Generator().manual_seed(3)
        sd: dict[str, torch.Tensor] = {}
        for b in range(2):
            base = f"transformer_blocks.{b}.attn.base_layer"
            sd[f"{base}.lora_A.weight"] = torch.randn(4, 16, generator=g)
            sd[f"{base}.lora_B.weight"] = torch.randn(16, 4, generator=g)
        lora = LoRA.from_state_dict(state_dict=sd)
        assert set(lora.targets) == {
            "transformer_blocks.0.attn.base_layer.weight",
            "transformer_blocks.1.attn.base_layer.weight",
        }

        s = _make_model_offloader(m, blocks_attr=["transformer_blocks"])
        _request_loras(s, [(lora, 0.5)], mode="routed")
        x = torch.randn(2, 16)
        _activate(s, torch.device("cpu"))
        try:
            actual = m(x).clone()
            # The hook lands on the matched ``.base_layer`` Linear.
            assert len(m.transformer_blocks[0].attn.base_layer._forward_hooks) == 1
        finally:
            s.deactivate()

        h = x
        for b in range(2):
            out = F.linear(h, base_w[b])
            a = sd[f"transformer_blocks.{b}.attn.base_layer.lora_A.weight"]
            bb = sd[f"transformer_blocks.{b}.attn.base_layer.lora_B.weight"]
            h = out + 0.5 * (h @ a.T) @ bb.T
        torch.testing.assert_close(actual, h, rtol=1e-4, atol=1e-4)

    def test_deactivate_releases_active_loras(self) -> None:
        m = _make_bf16_model(num_blocks=2, dim=16)
        lora = _make_lora(num_blocks=2, dim=16, seed=4)
        s = _make_model_offloader(m, blocks_attr=["transformer_blocks"])

        _request_loras(s, [(lora, 1.0)], mode="routed")
        _activate(s, torch.device("cpu"))
        # One streaming engine; one stacked hook per (LoRA, target).
        assert s._active_loras == [lora]
        assert len(s._lora_hook_handles) == 2

        s.deactivate()
        assert s._active_loras == []
        assert s._lora_hook_handles == []

        # A new activation-scoped request works after teardown.
        _request_loras(s, [(lora, 1.0)], mode="routed")
        _activate(s, torch.device("cpu"))
        try:
            assert len(s._active_loras) == 1
        finally:
            s.deactivate()

    @CUDA
    def test_routed_streaming_matches_merge(self) -> None:
        # A blocks_attr LoRA streams its factors co-scheduled with the model
        # (num_resident_blocks=1 forces streaming). Routed (resident-factor
        # residual) must match merge (weight-bake) bit-close, and the factor
        # holder modules must keep stable identity across stream load/evict
        # (the hook caches the node, not the churning .weight).
        m = _make_bf16_model(num_blocks=2, dim=16)
        lora = LoRA.from_state_dict(
            state_dict=_make_lora_sd(num_blocks=2, dim=16, seed=7),
            blocks_attr=["transformer_blocks"],
        )
        x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
        s = _make_model_offloader(m, blocks_attr=["transformer_blocks"])
        cfg = _strategy_stream_config(m, num_resident_blocks=1)

        _request_loras(s, [(lora, 0.5)], mode="merge")
        _activate(s, "cuda", stream_config=cfg)
        try:
            merged = m(x).clone()
            torch.cuda.synchronize()
        finally:
            s.deactivate()

        a_node_before, _ = lora.factor_holders(
            "transformer_blocks.0.attn.weight",
        )
        _request_loras(s, [(lora, 0.5)], mode="routed")
        _activate(s, "cuda", stream_config=cfg)
        try:
            routed = m(x).clone()
            torch.cuda.synchronize()
            a_node_after, _ = lora.factor_holders(
                "transformer_blocks.0.attn.weight",
            )
        finally:
            s.deactivate()

        assert a_node_before is a_node_after
        # Routed (resident-factor residual) vs merge (weight-bake) differ only
        # by bf16 rounding of the separate residual GEMMs.
        torch.testing.assert_close(routed, merged, rtol=0.1, atol=0.1)

    def test_activate_rolls_back_composite_when_routed_registration_fails(
        self,
    ) -> None:
        # Routed hooks install AFTER the composite is live. If registration
        # raises (here: a target the model doesn't manage), the now-resident
        # composite must be torn back down — not stranded. Proven by a clean
        # re-activation, which the composite's "already active" guard would
        # otherwise reject.
        m = _make_bf16_model(num_blocks=2, dim=16)
        sd = {
            "transformer_blocks.0.absent.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.absent.lora_B.weight": torch.randn(16, 4),
        }
        bad = LoRA.from_state_dict(
            state_dict=sd,
            blocks_attr=["transformer_blocks"],
        )
        s = _make_model_offloader(m, blocks_attr=["transformer_blocks"])
        _request_loras(s, [(bad, 1.0)], mode="routed")

        with pytest.raises(ValueError, match="not managed"):
            _activate(s, torch.device("cpu"))

        assert s.active_device is None
        _assert_lora_routing_available(bad)
        assert s._active_loras == []
        assert s._lora_hook_handles == []
        # Composite was deactivated -> a fresh activation succeeds and runs.
        _activate(s, torch.device("cpu"))
        try:
            out = m(torch.randn(2, 16, dtype=torch.bfloat16))
            assert out.shape == (2, 16)
        finally:
            s.deactivate()


class TestDeactivateCleanupInvariants:
    @CUDA
    def test_cleanup_runs_even_when_streamer_deactivate_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        _request_loras(s, [(_make_lora(4, 16), 1.0)])

        def streamer_boom() -> None:
            raise RuntimeError("streamer cleanup failed")

        monkeypatch.setattr(streamed_components(s)[0], "deactivate", streamer_boom)
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))

        with pytest.raises(RuntimeError):
            s.deactivate()

    def test_deactivate_tears_down_composite_even_if_lora_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A routed LoRA whose deactivate() raises (e.g. a factor
        # eviction CUDA error) must NOT prevent the base composite teardown —
        # otherwise the model would leak resident. CPU activation is enough to
        # exercise the lifecycle ordering.
        m = _make_bf16_model(num_blocks=2, dim=16)
        lora = _make_lora(num_blocks=2, dim=16, seed=4)
        s = _make_model_offloader(m, blocks_attr=["transformer_blocks"])
        _request_loras(s, [(lora, 1.0)], mode="routed")
        _activate(s, torch.device("cpu"))
        assert s._active_loras  # a live streaming engine to sabotage

        def boom() -> None:
            raise RuntimeError("LoRA eviction failed")

        monkeypatch.setattr(s._active_loras[0], "deactivate", boom)

        with pytest.raises(RuntimeError, match="LoRA eviction failed"):
            s.deactivate()

        # Despite the LoRA failure, the composite was deactivated and the
        # offloader reports inactive -> a fresh activation succeeds.
        assert s.active_device is None
        _activate(s, torch.device("cpu"))
        s.deactivate()


# ---------------------------------------------------------------------------
# Cache budget
# ---------------------------------------------------------------------------


class TestCacheBytes:
    def test_lora_cache_bytes_reports_factor_size(self) -> None:
        lora = _make_lora(num_blocks=4, dim=16, rank=4)
        assert lora.cache_bytes > 0


# ---------------------------------------------------------------------------
# Unified LoRA resource (ResourceCache integration)
# ---------------------------------------------------------------------------


class TestLoRAResource:
    def test_lora_is_cached_resource_with_activation_lifecycle(self) -> None:
        lora = _make_lora(num_blocks=2, dim=8, rank=2)
        assert isinstance(lora, ResourceStore)
        assert not isinstance(lora, ResourceBinding)
        assert not isinstance(lora, nn.Module)
        assert callable(lora.activate) and callable(lora.deactivate)

    def test_lora_through_resource_cache(self) -> None:
        sd = _make_lora_sd(num_blocks=2, dim=8, rank=2)
        cache = ResourceCache(10**9)
        spec = LoRASpec(
            key="lora:test",
            estimated_cache_bytes=1000,
            factory=lambda: sd,
        )
        with cache.lease(spec) as lora:
            assert isinstance(lora, LoRA)
            assert lora.cache_bytes > 0
            assert len(lora.targets) == 2
        with cache.lease("lora:test") as lora2:
            assert lora2 is lora

    def test_cached_model_runner_applies_loras_and_holds_leases(
        self,
    ) -> None:
        sd = _make_lora_sd(num_blocks=2, dim=8, rank=2, seed=17)
        expected_lora = LoRA.from_state_dict(sd)
        factory_calls = {"lora": 0}

        def lora_factory() -> dict[str, torch.Tensor]:
            factory_calls["lora"] += 1
            return sd

        cache = ResourceCache(10**9)
        runner = CachedModelRunner(cache)
        lora_spec = LoRASpec(
            key="lora:style",
            estimated_cache_bytes=1000,
            factory=lora_factory,
        )
        model_spec = ModelSpec(
            key="model",
            estimated_cache_bytes=10**6,
            factory=lambda: _make_bf16_model(num_blocks=2, dim=8).to(torch.float32),
        )

        x = torch.randn(2, 8)
        with runner.use(
            model_spec,
            device="cpu",
            lora_specs=[lora_spec],
            lora_strengths=[0.5],
            lora_mode="routed",
        ) as model:
            assert cache.info("lora:style").lease_count == 1
            assert cache.info("model").lease_count == 1
            actual = model(x)
            expected = _expected_routed_output(model, x, [(expected_lora, 0.5)])
            assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)

        assert cache.info("lora:style").lease_count == 0
        assert cache.info("model").lease_count == 0
        assert factory_calls["lora"] == 1

        with runner.use(
            model_spec,
            device="cpu",
            lora_specs=[lora_spec],
            lora_mode="routed",
        ) as model:
            assert cache.info("lora:style").lease_count == 1
            actual = model(x)
            expected = _expected_routed_output(model, x, [(expected_lora, 1.0)])
            assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
        assert factory_calls["lora"] == 1

    def test_cached_lora_rejects_overlap_across_model_runtimes(self) -> None:
        cache = ResourceCache(10**9)
        runner = CachedModelRunner(cache)
        lora_spec = LoRASpec(
            key="lora:shared",
            estimated_cache_bytes=1000,
            factory=lambda: _make_lora_sd(num_blocks=2, dim=8, rank=2),
        )
        first_spec = ModelSpec(
            key="model:first",
            estimated_cache_bytes=10**6,
            factory=lambda: _make_bf16_model(2, 8).to(torch.float32),
        )
        second_spec = ModelSpec(
            key="model:second",
            estimated_cache_bytes=10**6,
            factory=lambda: _make_bf16_model(2, 8).to(torch.float32),
        )
        x = torch.randn(2, 8)

        with runner.use(
            first_spec,
            device="cpu",
            lora_specs=[lora_spec],
            lora_mode="routed",
        ) as first:
            with pytest.raises(LoRARuntimeInUseError, match="active routed use"):
                with runner.use(
                    second_spec,
                    device="cpu",
                    lora_specs=[lora_spec],
                    lora_mode="routed",
                ):
                    pass
            assert cache.info("lora:shared").lease_count == 1
            assert first(x).shape == (2, 8)

        with runner.use(
            second_spec,
            device="cpu",
            lora_specs=[lora_spec],
            lora_mode="routed",
        ) as second:
            assert second(x).shape == (2, 8)

    def test_runner_rejects_strength_mismatch_before_cache_admission(self) -> None:
        factory_calls = {"lora": 0, "model": 0}

        def lora_factory() -> dict[str, torch.Tensor]:
            factory_calls["lora"] += 1
            return _make_lora_sd(num_blocks=2, dim=8, rank=2)

        def model_factory() -> nn.Module:
            factory_calls["model"] += 1
            return _make_bf16_model(num_blocks=2, dim=8)

        lora_spec = LoRASpec(
            key="lora:style",
            estimated_cache_bytes=1000,
            factory=lora_factory,
        )
        model_spec = ModelSpec(
            key="model",
            estimated_cache_bytes=10**6,
            factory=model_factory,
        )
        cache = ResourceCache(10**9)
        runner = CachedModelRunner(cache)

        with pytest.raises(ValueError, match="same length"):
            with runner.use(
                model_spec,
                device="cpu",
                lora_specs=[lora_spec],
                lora_strengths=[],
            ):
                pass

        assert factory_calls == {"lora": 0, "model": 0}
        with pytest.raises(ResourceNotRegisteredError):
            cache.info("lora:style")
        with pytest.raises(ResourceNotRegisteredError):
            cache.info("model")

    def test_runner_rejects_duplicate_lora_before_cache_admission(self) -> None:
        factory_calls = {"lora": 0, "model": 0}

        def lora_factory() -> dict[str, torch.Tensor]:
            factory_calls["lora"] += 1
            return _make_lora_sd(num_blocks=2, dim=8, rank=2)

        def model_factory() -> nn.Module:
            factory_calls["model"] += 1
            return _make_bf16_model(num_blocks=2, dim=8)

        lora_spec = LoRASpec(
            key="lora:style",
            estimated_cache_bytes=1000,
            factory=lora_factory,
        )
        model_spec = ModelSpec(
            key="model",
            estimated_cache_bytes=10**6,
            factory=model_factory,
        )
        cache = ResourceCache(10**9)
        runner = CachedModelRunner(cache)

        with pytest.raises(ValueError, match="same LoRA resource key"):
            with runner.use(
                model_spec,
                device="cpu",
                lora_specs=[lora_spec, lora_spec],
                lora_mode="routed",
            ):
                pass

        assert factory_calls == {"lora": 0, "model": 0}

    def test_lora_leased_during_resource_cache_miss(self) -> None:
        # The LoRAs acquired by a runner use are leased before the
        # model store builds, so a model cache-miss must fail admission
        # rather than evict a lora it is about to be applied with.
        lora_spec = LoRASpec(
            key="lora:style",
            estimated_cache_bytes=1000,
            factory=lambda: _make_lora_sd(num_blocks=2, dim=8, rank=2),
        )
        model_spec = ModelSpec(
            key="model",
            estimated_cache_bytes=10_000,
            factory=lambda: _make_bf16_model(num_blocks=2, dim=8).to(torch.float32),
        )

        cache = ResourceCache(10_000)
        runner = CachedModelRunner(cache)
        with pytest.raises(ResourceTooLargeError):
            with runner.use(
                model_spec,
                device="cpu",
                lora_specs=[lora_spec],
                lora_mode="routed",
            ):
                pass

        assert cache.info("lora:style").cached
        assert cache.info("lora:style").lease_count == 0


# ---------------------------------------------------------------------------
# LoRA-owned module + composite store (blocks_attr)
# ---------------------------------------------------------------------------


class TestLoRABlocksAttr:
    """LoRA factors live in an nn.Module pinned via a CompositeComponentStore;
    blocks_attr groups factor blocks into a streamed component."""

    def test_targets_match_all_pinned(self) -> None:
        # A blocks_attr-built LoRA derives the same targets (keys + factor
        # bytes + cache_bytes) as the default all-pinned one, so it merges
        # identically — the transitive guarantee for the merge suite.
        sd = _make_lora_sd(num_blocks=4, dim=16, seed=7)
        pinned = LoRA.from_state_dict(state_dict=sd)
        streamed = LoRA.from_state_dict(state_dict=sd, blocks_attr=["transformer_blocks"])
        assert set(pinned.targets) == set(streamed.targets)
        for key in pinned.targets:
            pa, pb = _factor_tensors(pinned.targets[key])
            sa, sb = _factor_tensors(streamed.targets[key])
            torch.testing.assert_close(pa, sa)
            torch.testing.assert_close(pb, sb)
        assert pinned.cache_bytes == streamed.cache_bytes

    def test_scaled_factor_views_pinned_memory(self) -> None:
        # The derived factor's .scaled() must view pinned host bytes, or
        # merge's non_blocking H2D silently degrades to a sync stall.
        sd = _make_lora_sd(num_blocks=2, dim=16)
        lora = LoRA.from_state_dict(state_dict=sd, blocks_attr=["transformer_blocks"])
        factor = next(iter(lora.targets.values()))
        scaled = factor.scaled(1.0)
        assert scaled.a.is_pinned()
        assert scaled.b.is_pinned()

    def test_blocks_attr_builds_streamed_component(self) -> None:
        sd = _make_lora_sd(num_blocks=4, dim=16)
        lora = LoRA.from_state_dict(state_dict=sd, blocks_attr=["transformer_blocks"])
        streamed = lora._composite_store.streamed_stores
        assert len(streamed) == 1
        assert streamed[0].blocks_path == "transformer_blocks"
        # Default (no blocks_attr) keeps everything pinned, no streamed group.
        assert LoRA.from_state_dict(state_dict=sd)._composite_store.streamed_stores == ()

    @staticmethod
    def _sparse_sd() -> dict[str, torch.Tensor]:
        # Adapts only blocks 0 and 2 — block 1 is unadapted.
        g = torch.Generator().manual_seed(0)
        sd: dict[str, torch.Tensor] = {}
        for b in (0, 2):
            base = f"transformer_blocks.{b}.attn"
            sd[f"{base}.lora_A.weight"] = torch.randn(4, 16, generator=g)
            sd[f"{base}.lora_B.weight"] = torch.randn(16, 4, generator=g)
        return sd

    def test_path_walk_indexes_sparse_blocks_with_empty_holders(self) -> None:
        # Path-walk construction indexes blocks truly: the gap at block 1
        # becomes an empty holder so blocks 0 and 2 keep their real indices.
        # No manual padding — it falls out of the dotted paths.
        lora = LoRA.from_state_dict(state_dict=self._sparse_sd())
        blocks = lora._module.transformer_blocks  # type: ignore[union-attr]
        assert len(blocks) == 3
        assert list(blocks[1].parameters()) == []  # empty holder at the gap
        assert set(lora.targets) == {
            "transformer_blocks.0.attn.weight",
            "transformer_blocks.2.attn.weight",
        }

    def test_nested_module_list_paths(self) -> None:
        # Consecutive numeric segments build a ModuleList-of-ModuleLists: the
        # holder type follows whether the *next* segment is an index, so nested
        # lists (e.g. grouped/MoE blocks at ``0.1``) round-trip correctly.
        g = torch.Generator().manual_seed(0)
        sd = {
            "blocks.0.1.attn.lora_A.weight": torch.randn(4, 16, generator=g),
            "blocks.0.1.attn.lora_B.weight": torch.randn(16, 4, generator=g),
        }
        lora = LoRA.from_state_dict(state_dict=sd)
        assert set(lora.targets) == {"blocks.0.1.attn.weight"}
        blocks = lora._module.blocks  # type: ignore[union-attr]
        assert isinstance(blocks, nn.ModuleList)
        assert isinstance(blocks[0], nn.ModuleList)
        # The path the routed apply resolves on the model is reachable.
        assert lora.factor_holders("blocks.0.1.attn.weight")[0] is not None

    def test_root_level_numeric_paths(self) -> None:
        # A root-level nn.Sequential / nn.ModuleList model has params like
        # ``0.weight``, so factor keys start with an index — the root holder
        # must itself be a ModuleList (regression: a plain-Module root crashed
        # the path-walk on the first numeric segment).
        g = torch.Generator().manual_seed(0)
        sd = {
            "0.lora_A.weight": torch.randn(4, 16, generator=g),
            "0.lora_B.weight": torch.randn(16, 4, generator=g),
        }
        lora = LoRA.from_state_dict(state_dict=sd)
        assert set(lora.targets) == {"0.weight"}
        assert isinstance(lora._module, nn.ModuleList)

    def test_sparse_blocks_attr_streaming_skips_empty_holders(self) -> None:
        # A sparse LoRA streams fine now (#26): the empty holder block at the
        # gap is skipped, the two adapted blocks pass the uniform-name check,
        # and their TRUE indices are recorded so the seam can co-schedule them
        # against the matching base-model blocks.
        lora = LoRA.from_state_dict(
            state_dict=self._sparse_sd(),
            blocks_attr=["transformer_blocks"],
        )
        (streamed,) = lora._composite_store.streamed_stores
        assert streamed.block_indices == (0, 2)
        assert len(streamed._block_stores) == 2
        # Names use TRUE indices (0, 2), so there is no bogus block-1 target
        # and no duplicate block-2 target leaking through the pinned remainder.
        assert streamed.param_names == frozenset(
            {
                "transformer_blocks.0.attn.lora_A.weight",
                "transformer_blocks.0.attn.lora_B.weight",
                "transformer_blocks.2.attn.lora_A.weight",
                "transformer_blocks.2.attn.lora_B.weight",
            }
        )
        assert set(lora.targets) == {
            "transformer_blocks.0.attn.weight",
            "transformer_blocks.2.attn.weight",
        }

    def test_non_block_adapter_goes_to_pinned_remainder(self) -> None:
        sd = _make_lora_sd(num_blocks=2, dim=16)
        sd["embed.lora_A.weight"] = torch.randn(4, 16)
        sd["embed.lora_B.weight"] = torch.randn(16, 4)
        lora = LoRA.from_state_dict(state_dict=sd, blocks_attr=["transformer_blocks"])
        assert lora._composite_store.pinned_store is not None
        assert "embed.weight" in lora.targets
        assert "transformer_blocks.0.attn.weight" in lora.targets

    @CUDA
    def test_blocks_attr_merge_matches_manual_baseline(self) -> None:
        # End-to-end: a streamed-store-pinned LoRA merges correctly, proving
        # the derived factors keep pinned-view async H2D semantics.
        m = _make_bf16_model(num_blocks=4, dim=16)
        captured = {i: m.transformer_blocks[i].attn.weight.detach().clone() for i in range(4)}
        sd = _make_lora_sd(num_blocks=4, dim=16, seed=3)
        lora = LoRA.from_state_dict(state_dict=sd, blocks_attr=["transformer_blocks"])
        s = _make_strategy(m)
        _request_loras(s, [(lora, 0.5)])
        _activate(s, "cuda", stream_config=_strategy_stream_config(m))
        try:
            for i in range(4):
                _ = m.transformer_blocks[i](torch.randn(2, 16, dtype=torch.bfloat16, device="cuda"))
                torch.cuda.synchronize()
                expected = _expected_merged_weight(
                    captured[i].to("cuda"),
                    [(lora, 0.5)],
                    i,
                    "attn.weight",
                )
                actual = m.transformer_blocks[i].attn.weight.detach()
                assert torch.allclose(actual, expected, rtol=0.01, atol=0.01)
        finally:
            s.deactivate()


# ---------------------------------------------------------------------------
# Routed LoRA activation lifecycle
# ---------------------------------------------------------------------------


class TestLoRAActivation:
    def test_cpu_activation_round_trips_and_rejects_overlap(self) -> None:
        lora = _make_lora(num_blocks=4, dim=16)
        model = _make_bf16_model(num_blocks=4, dim=16)
        lora.activate(
            torch.device("cpu"),
            schedule_model=model,
        )
        try:
            assert lora._composite is not None
            with pytest.raises(LoRARuntimeInUseError, match="active routed use"):
                lora.activate(
                    torch.device("cpu"),
                    schedule_model=model,
                )
        finally:
            lora.deactivate()
        _assert_lora_routing_available(lora)

    def test_activate_rejects_unsupported_device(self) -> None:
        # Device-type validation is delegated to the offload components, which
        # support CUDA/CPU only — MPS (and anything else) is rejected there
        # (no MPS backend needed: rejection is by device.type before any move).
        lora = _make_lora(num_blocks=2, dim=8)
        with pytest.raises(ValueError, match="supports CUDA or CPU"):
            lora.activate(
                torch.device("mps"),
                schedule_model=_make_bf16_model(num_blocks=2, dim=8),
            )
        _assert_lora_routing_available(lora)

    def test_deactivate_releases_claim_even_if_teardown_raises(
        self,
    ) -> None:
        # A teardown error must not leave the resource claim wedged.
        lora = _make_lora(num_blocks=2, dim=8)
        lora.activate(
            torch.device("cpu"),
            schedule_model=_make_bf16_model(num_blocks=2, dim=8),
        )

        def boom() -> None:
            raise RuntimeError("teardown failed")

        assert lora._composite is not None
        lora._composite.deactivate = boom  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="teardown failed"):
            lora.deactivate()
        _assert_lora_routing_available(lora)

    def test_two_model_offloaders_cannot_share_active_lora(self) -> None:
        lora = _make_lora(num_blocks=2, dim=8)
        first = _make_model_offloader(_make_bf16_model(2, 8).to(torch.float32))
        second = _make_model_offloader(_make_bf16_model(2, 8).to(torch.float32))

        first.activate("cpu", loras=[lora], lora_mode="routed")
        try:
            with pytest.raises(LoRARuntimeInUseError, match="active routed use"):
                second.activate("cpu", loras=[lora], lora_mode="routed")
            assert second.active_device is None
            assert first.value(torch.randn(2, 8)).shape == (2, 8)
        finally:
            first.deactivate()

        second.activate("cpu", loras=[lora], lora_mode="routed")
        second.deactivate()

    def test_lora_conflict_rolls_back_preceding_lora_and_model(self) -> None:
        available = _make_lora(num_blocks=2, dim=8, seed=1)
        busy = _make_lora(num_blocks=2, dim=8, seed=2)
        offloader = _make_model_offloader(
            _make_bf16_model(2, 8).to(torch.float32),
        )

        busy_model = _make_bf16_model(2, 8)
        busy.activate("cpu", schedule_model=busy_model)
        try:
            with pytest.raises(LoRARuntimeInUseError, match="active routed use"):
                offloader.activate(
                    "cpu",
                    loras=[available, busy],
                    lora_mode="routed",
                )
            _assert_lora_routing_available(available)
            with pytest.raises(LoRARuntimeInUseError, match="active routed use"):
                busy.activate("cpu", schedule_model=busy_model)
            assert offloader.active_device is None
        finally:
            busy.deactivate()

        offloader.activate("cpu")
        offloader.deactivate()

    @CUDA
    def test_factor_blocks_stream_co_scheduled_with_model(self) -> None:
        # The factor block for adapter slot i lands on GPU when the MATCHING
        # base-model block runs its forward (schedule_model co-scheduling),
        # bounded by the residency budget, and the streamed bytes equal the
        # pinned master. This is the resident streaming a future routed path
        # reads off the module.
        dim, num_blocks = 16, 4
        lora = LoRA.from_state_dict(
            _make_lora_sd(num_blocks=num_blocks, dim=dim, seed=9),
            blocks_attr=["transformer_blocks"],
        )
        model = _make_bf16_model(num_blocks=num_blocks, dim=dim).to("cuda")
        factor_holders = [
            lora.factor_holders(
                f"transformer_blocks.{i}.attn.weight"
            )
            for i in range(num_blocks)
        ]
        cpu_factors = [
            (
                a.get_parameter("weight").detach().clone(),
                b.get_parameter("weight").detach().clone(),
            )
            for a, b in factor_holders
        ]

        lora.activate(
            torch.device("cuda"),
            schedule_model=model,
            stream_config=StreamConfig(
                num_resident_blocks=1,
                num_prefetch_blocks=1,
            ),
        )
        try:
            x = torch.randn(2, dim, dtype=torch.bfloat16, device="cuda")
            for i in range(num_blocks):
                model.transformer_blocks[i](x)
                torch.cuda.synchronize()
                a = factor_holders[i][0].get_parameter("weight")
                b = factor_holders[i][1].get_parameter("weight")
                assert a.is_cuda and b.is_cuda
                torch.testing.assert_close(a.cpu(), cpu_factors[i][0])
                torch.testing.assert_close(b.cpu(), cpu_factors[i][1])
        finally:
            lora.deactivate()

        # Teardown returns every factor block to pinned CPU.
        for a, b in factor_holders:
            assert not a.get_parameter("weight").is_cuda
            assert not b.get_parameter("weight").is_cuda
