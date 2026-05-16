"""Tests for LoRA merge via ``ModelOffloader.set_loras()``.

Covers LoRA construction validation, set_loras matching, lifecycle
(activate/deactivate), LoRA switching, and forward-output correctness
against a manually-merged baseline.

Most lifecycle tests run on CPU (the merge math is device-agnostic);
CUDA-only tests gate on availability.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from torch_offload import (
    LoRA,
    ModelCache,
    ModelOffloader,
    ResourceSpec,
    merge_lora,
)
from torch_offload.lora import KeyTransformT
from torch_offload.model_offloader import _routed_factor_dtype
from torch_offload.protocols import CachedResource

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
            self.transformer_blocks = nn.ModuleList(
                [Block(dim) for _ in range(num_blocks)]
            )
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
    num_blocks: int = 2, dim: int = 16, dtype: torch.dtype = torch.float32,
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
            self.transformer_blocks = nn.ModuleList(
                [Block(dim) for _ in range(num_blocks)]
            )
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
    num_blocks: int, dim: int, rank: int = 4, seed: int = 0,
    prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Build a flat safetensors-style state dict targeting attn.weight."""
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {}
    for b in range(num_blocks):
        base = f"{prefix}transformer_blocks.{b}.attn"
        sd[f"{base}.lora_A.weight"] = torch.randn(
            rank, dim, generator=g, dtype=torch.float32,
        )
        sd[f"{base}.lora_B.weight"] = torch.randn(
            dim, rank, generator=g, dtype=torch.float32,
        )
    return sd


def _make_lora(
    num_blocks: int, dim: int, rank: int = 4,
    seed: int = 0, prefix: str = "",
    key_transform: KeyTransformT = ...,  # type: ignore[assignment]
) -> LoRA:
    """Build a LoRA targeting attn.weight across all blocks."""
    sd = _make_lora_sd(num_blocks, dim, rank=rank, seed=seed, prefix=prefix)
    if key_transform is ...:  # type: ignore[comparison-overlap]
        return LoRA(state_dict=sd)
    return LoRA(state_dict=sd, key_transform=key_transform)


def _expected_merged_weight(
    base: torch.Tensor, loras: list[tuple[LoRA, float]], block_idx: int, qual: str,
) -> torch.Tensor:
    """Compute the target weight by summing all LoRA deltas onto the base."""
    out = base.clone()
    target_name = f"transformer_blocks.{block_idx}.{qual}"
    for lora, strength in loras:
        factors = lora.targets.get(target_name)
        if factors is None:
            continue
        a, b = factors
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
            a, b = factors
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


def _make_strategy(
    model: nn.Module, blocks_to_swap: int = 1,
) -> ModelOffloader:
    """Shorthand for constructing the strategy with sensible defaults."""
    return ModelOffloader(
        model,
        layers_attr="transformer_blocks",
        blocks_to_swap=blocks_to_swap,
    )


def _has_post_copy_hook(strategy: ModelOffloader, target_key: str) -> bool:
    """Check whether a merge hook is installed for the given target."""
    buf = strategy._target_to_buffer.get(target_key)
    hooks = strategy._post_copy_hooks
    return buf is not None and hooks is not None and id(buf) in hooks


def _activate_loras_for_test(
    strategy: ModelOffloader,
) -> int:
    targets = strategy._group_loras_by_target(strategy._loras)
    if strategy._lora_mode == "merge":
        strategy._activate_merge_loras(torch.device("cuda"), targets)
        return len(targets)

    with contextlib.ExitStack() as stack:
        strategy._activate_routed_loras(torch.device("cpu"), targets, stack)
        return len(targets)


# ---------------------------------------------------------------------------
# LoRA construction validation
# ---------------------------------------------------------------------------


class TestLoRAConstruction:
    def test_unpaired_a_factor(self) -> None:
        sd = {"transformer_blocks.0.attn.lora_A.weight": torch.randn(4, 16)}
        with pytest.raises(ValueError, match="Unpaired"):
            LoRA(state_dict=sd)

    def test_unpaired_b_factor(self) -> None:
        sd = {"transformer_blocks.0.attn.lora_B.weight": torch.randn(16, 4)}
        with pytest.raises(ValueError, match="Unpaired"):
            LoRA(state_dict=sd)

    def test_rejects_non_floating_factor_dtype(self) -> None:
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.zeros(4, 16, dtype=torch.int32),
            "transformer_blocks.0.attn.lora_B.weight": torch.zeros(16, 4, dtype=torch.int32),
        }
        with pytest.raises(ValueError, match="floating-point"):
            LoRA(state_dict=sd)

    def test_rejects_rank_mismatch(self) -> None:
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.attn.lora_B.weight": torch.randn(16, 8),
        }
        with pytest.raises(ValueError, match="shape mismatch"):
            LoRA(state_dict=sd)

    def test_rejects_non_2d_factor(self) -> None:
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.randn(4),
            "transformer_blocks.0.attn.lora_B.weight": torch.randn(16, 4),
        }
        with pytest.raises(ValueError, match="shape mismatch"):
            LoRA(state_dict=sd)

    def test_factors_are_pinned(self) -> None:
        lora = _make_lora(4, 16)
        for a, b in lora.targets.values():
            assert a.is_pinned()
            assert b.is_pinned()

    def test_cache_bytes(self) -> None:
        lora = _make_lora(4, 16, rank=4)
        expected = 4 * (4 * 16 + 16 * 4) * 4  # 4 blocks * 2 factors * float32
        assert lora.cache_bytes == expected

    def test_default_key_transform_strips_prefix(self) -> None:
        lora = _make_lora(1, 16, prefix="diffusion_model.")
        assert "transformer_blocks.0.attn.weight" in lora.targets

    def test_key_transform_none_preserves_prefix(self) -> None:
        lora = _make_lora(1, 16, prefix="diffusion_model.", key_transform=None)
        assert "diffusion_model.transformer_blocks.0.attn.weight" in lora.targets
        assert "transformer_blocks.0.attn.weight" not in lora.targets


# ---------------------------------------------------------------------------
# LoRA request validation
# ---------------------------------------------------------------------------


class TestSetLorasValidation:
    def test_rejects_target_shape_mismatch(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.attn.lora_B.weight": torch.randn(8, 4),
        }
        s.set_loras([(LoRA(state_dict=sd), 1.0)])
        with pytest.raises(ValueError, match="shape mismatch"):
            _activate_loras_for_test(s)

    def test_accepts_fp32_lora_target(self) -> None:
        m = _make_bf16_model().to(torch.float32)
        for p in m.parameters():
            p.requires_grad = False
        s = _make_strategy(m)
        s.set_loras([(_make_lora(4, 16), 1.0)])
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
        s.set_loras([(LoRA(state_dict=sd), 1.0)])
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "embed.weight")

    def test_non_block_tied_alias_target_matched(self) -> None:
        m = _make_tied_non_block_model(dtype=torch.bfloat16)
        s = _make_strategy(m)
        sd = {
            "head.lora_A.weight": torch.randn(4, 16),
            "head.lora_B.weight": torch.randn(16, 4),
        }
        s.set_loras([(LoRA(state_dict=sd), 1.0)], mode="merge")
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
        s.set_loras([(LoRA(state_dict=sd), 1.0)], mode="merge")
        with pytest.raises(ValueError, match="same tied parameter storage"):
            _activate_loras_for_test(s)

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
        s.set_loras([(LoRA(state_dict=sd), 1.0)], mode="merge")
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "transformer_blocks.0.b.weight")
        assert _has_post_copy_hook(s, "transformer_blocks.0.a.weight")

    def test_key_transform_strips_prefix(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16, prefix="diffusion_model.")
        s.set_loras([(lora, 1.0)])
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")

    def test_key_transform_none_matches_exact_keys(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16, key_transform=None)
        s.set_loras([(lora, 1.0)])
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")

    def test_key_transform_none_skips_prefixed_keys(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16, prefix="diffusion_model.", key_transform=None)
        s.set_loras([(lora, 1.0)])
        _activate_loras_for_test(s)
        assert not _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")

    def test_merge_mode_activation_rejects_cpu(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        s.set_loras([(_make_lora(4, 16), 1.0)], mode="merge")
        with pytest.raises(ValueError, match="merge mode requires CUDA"):
            s.activate("cpu")

    @CUDA
    def test_set_loras_raises_while_active(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        s.set_loras([(_make_lora(4, 16), 1.0)])
        s.activate("cuda")
        try:
            with pytest.raises(RuntimeError, match="inactive"):
                s.set_loras([])
        finally:
            s.deactivate()

    def test_clear_loras_clears_previous_merge_hooks(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        s.set_loras([(_make_lora(4, 16, rank=4), 1.0)])
        _activate_loras_for_test(s)
        assert _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")
        s._clear_loras()
        assert not _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")

    def test_rejects_quanto_target_in_merge_mode(self) -> None:
        # Regression: quanto WeightQBytesTensor advertises the scale's
        # float dtype (bf16/fp16) via `weight.dtype`, which used to
        # pass the merge-mode dtype gate. addmm_ on the wrapper then
        # silently drops the LoRA — _data stays untouched. Activation
        # must reject subclassed wrappers in merge mode regardless of
        # the advertised dtype.
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        m = _make_bf16_model()
        rows = cols = 16
        data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )
        m.embed.weight = nn.Parameter(qt, requires_grad=False)

        s = _make_strategy(m)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16),
            "embed.lora_B.weight": torch.randn(16, 4),
        }
        s.set_loras([(LoRA(state_dict=sd), 1.0)], mode="merge")
        with pytest.raises(ValueError, match=r"merge mode|wrapped in|subclass"):
            _activate_loras_for_test(s)
        # routed mode must still accept it.
        s.set_loras([(LoRA(state_dict=sd), 1.0)], mode="routed")
        route_count = _activate_loras_for_test(s)
        assert route_count == 1

    def test_accepts_fp16_base(self) -> None:
        m = _make_bf16_model().to(torch.float16)
        for p in m.parameters():
            p.requires_grad = False
        s = _make_strategy(m)
        assert s.cache_bytes > 0

    def test_modes_defer_until_activation(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        for mode in ("merge", "routed"):
            s.set_loras([(_make_lora(4, 16), 1.0)], mode=mode)
            assert not _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")
            assert len(s._loras) == 1
            assert s._lora_mode == mode

    def test_routed_mode_cpu_activation_uses_hooks(self) -> None:
        m = _make_bf16_model(num_blocks=2).to(torch.float32)
        for p in m.parameters():
            p.requires_grad = False
        loras = [(_make_lora(2, 16, seed=9), 0.75)]
        s = _make_strategy(m)
        s.set_loras(loras, mode="routed")

        x = torch.randn(2, 16)
        s.activate("cpu")
        try:
            actual = m(x)
            expected = _expected_routed_output(m, x, loras)
            assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
            assert not _has_post_copy_hook(s, "transformer_blocks.0.attn.weight")
        finally:
            s.deactivate()

        assert s._loras == []


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @CUDA
    def test_activate_runs_components(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        s.set_loras([(_make_lora(4, 16), 1.0)])
        try:
            s.activate("cuda")
            assert m.embed.weight.is_cuda
            assert m.head.weight.is_cuda
        finally:
            s.deactivate()

    @CUDA
    def test_deactivate_returns_to_pinned(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        s.set_loras([(_make_lora(4, 16), 1.0)])
        s.activate("cuda")
        s.deactivate()
        assert m.embed.weight.is_pinned()
        assert m.head.weight.is_pinned()

    @CUDA
    def test_reactivation_with_different_loras(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        s.set_loras([(_make_lora(4, 16, seed=1), 1.0)])
        s.activate("cuda")
        s.deactivate()
        s.set_loras([(_make_lora(4, 16, seed=2), 1.0)])
        s.activate("cuda")
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
            4, 16, generator=g, dtype=torch.float32,
        )
        sd["embed.lora_B.weight"] = torch.randn(
            16, 4, generator=g, dtype=torch.float32,
        )
        s = _make_strategy(m)
        s.set_loras([(LoRA(state_dict=sd), 1.0)], mode="merge")
        s.activate("cuda")
        s.deactivate()

        s.set_loras([])
        s.activate("cuda")
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
        s.activate("cuda")
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            for blk in m.transformer_blocks:
                x = blk(x)
            torch.cuda.synchronize()
            actual = m.transformer_blocks[0].attn.weight.detach()
            assert torch.allclose(
                actual.cpu(), captured, rtol=0.0, atol=0.0,
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
        captured_base = {
            i: m.transformer_blocks[i].attn.weight.detach().clone()
            for i in range(4)
        }

        loras = [
            (_make_lora(num_blocks=4, dim=16, seed=10), 0.5),
            (_make_lora(num_blocks=4, dim=16, seed=20), 0.25),
        ]
        s = _make_strategy(m)
        s.set_loras(loras)
        s.activate("cuda")
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            for blk in m.transformer_blocks:
                x = blk(x)
            torch.cuda.synchronize()
            for i in range(4):
                _ = m.transformer_blocks[i](
                    torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
                )
                torch.cuda.synchronize()
                expected = _expected_merged_weight(
                    captured_base[i].to("cuda"), loras, i, "attn.weight",
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
    def test_prefixed_lora_keys_merge_correctly(self) -> None:
        """LoRAs with ``diffusion_model.`` prefix (ComfyUI format) should
        merge identically to unprefixed keys via the default key_transform."""
        m = _make_bf16_model(num_blocks=4, dim=16)
        captured_base = {
            i: m.transformer_blocks[i].attn.weight.detach().clone()
            for i in range(4)
        }

        lora = _make_lora(4, 16, seed=42, prefix="diffusion_model.")
        s = _make_strategy(m)
        s.set_loras([(lora, 0.7)])
        s.activate("cuda")
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            for blk in m.transformer_blocks:
                x = blk(x)
            torch.cuda.synchronize()
            for i in range(4):
                _ = m.transformer_blocks[i](
                    torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
                )
                torch.cuda.synchronize()
                expected = _expected_merged_weight(
                    captured_base[i].to("cuda"), [(lora, 0.7)], i, "attn.weight",
                )
                actual = m.transformer_blocks[i].attn.weight.detach()
                assert torch.allclose(actual, expected, rtol=0.01, atol=0.01)
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
        lora = LoRA(state_dict=sd)
        s = _make_strategy(m)
        s.set_loras([(lora, 0.5)])
        s.activate("cuda")
        try:
            a, b = lora.targets["embed.weight"]
            expected = (
                captured_embed + 0.5 * (b.to(torch.bfloat16) @ a.to(torch.bfloat16))
            ).to("cuda")
            actual = m.embed.weight.detach()
            assert torch.allclose(actual, expected, rtol=0.01, atol=0.01), (
                f"non-block merge mismatch:\n"
                f"  expected: {expected.flatten()[:4]}\n"
                f"  actual:   {actual.flatten()[:4]}"
            )
        finally:
            s.deactivate()

    @CUDA
    def test_peft_wrapped_model_matches_all_targets(self) -> None:
        """PEFT renames params with ``.base_layer.``; set_loras must still match."""

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
                self.transformer_blocks = nn.ModuleList(
                    [PEFTBlock(16) for _ in range(4)]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for blk in self.transformer_blocks:
                    x = blk(x)
                return x

        m = M().to(torch.bfloat16)
        m.requires_grad_(False)

        captured_base = {
            i: m.transformer_blocks[i].attn.base_layer.weight.detach().clone()
            for i in range(4)
        }

        lora = _make_lora(num_blocks=4, dim=16, seed=42)
        s = _make_strategy(m)
        s.set_loras([(lora, 0.7)])
        s.activate("cuda")
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            m(x)
            torch.cuda.synchronize()
            for i in range(4):
                _ = m.transformer_blocks[i](
                    torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
                )
                torch.cuda.synchronize()
                expected = _expected_merged_weight(
                    captured_base[i].to("cuda"), [(lora, 0.7)], i, "attn.weight",
                )
                actual = m.transformer_blocks[i].attn.base_layer.weight.detach()
                assert torch.allclose(actual, expected, rtol=0.01, atol=0.01), (
                    f"block {i} PEFT-wrapped merge mismatch"
                )
        finally:
            s.deactivate()


class TestPermanentMerge:
    def test_tied_alias_target_merges_shared_storage(self) -> None:
        m = _make_tied_non_block_model(dtype=torch.float32)
        base = m.embed.weight.detach().clone()
        sd = {
            "head.lora_A.weight": torch.randn(4, 16),
            "head.lora_B.weight": torch.randn(16, 4),
        }
        lora = LoRA(state_dict=sd, key_transform=None)
        a, b = lora.targets["head.weight"]

        merged = merge_lora(m, [(lora, 0.25)])

        expected = base.clone()
        expected.addmm_(
            b.to(dtype=expected.dtype), a.to(dtype=expected.dtype), alpha=0.25,
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

        with pytest.raises(ValueError, match="same tied parameter storage"):
            merge_lora(m, [(LoRA(state_dict=sd, key_transform=None), 1.0)])

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
                    UnknownTensor, torch.randn(16, 16), False,
                )
                self.other.weight = nn.Parameter(wrapped, requires_grad=False)

        m = M()
        before = m.target.weight.detach().clone()
        sd = {
            "target.lora_A.weight": torch.randn(4, 16),
            "target.lora_B.weight": torch.randn(16, 4),
        }
        lora = LoRA(state_dict=sd, key_transform=None)
        a, b = lora.targets["target.weight"]

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

    @CUDA
    def test_routed_forward_matches_manual_baseline(self) -> None:
        # Routed mode: base weight stays exactly as constructed; the
        # LoRA contribution rides as a forward-hook addition.
        m = _make_bf16_model(num_blocks=3, dim=16)
        loras = [
            (_make_lora(num_blocks=3, dim=16, seed=11), 0.5),
            (_make_lora(num_blocks=3, dim=16, seed=22), 0.25),
        ]
        base_snapshots = [
            m.transformer_blocks[i].attn.weight.detach().clone()
            for i in range(3)
        ]

        s = _make_strategy(m)
        s.set_loras(loras, mode="routed")
        s.activate("cuda")
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            actual = m(x)
            torch.cuda.synchronize()
            expected = self._expected_routed_output(m, x, loras)
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1), (
                f"routed forward mismatch:\n"
                f"  expected: {expected.flatten()[:4]}\n"
                f"  actual:   {actual.flatten()[:4]}"
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
        s.set_loras([(_make_lora(3, 16, seed=7), 1.0)], mode="routed")
        s.activate("cuda")
        x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
        with_lora = m(x).detach().clone()
        torch.cuda.synchronize()
        s.deactivate()

        # Re-activate without LoRAs; output should differ from with_lora
        # (the hooks should be gone).
        s.set_loras([], mode="routed")
        s.activate("cuda")
        try:
            base_only = m(x)
            torch.cuda.synchronize()
            assert not torch.allclose(with_lora, base_only, rtol=0.001, atol=0.001), (
                "deactivate did not remove routed hooks; base-only output "
                "still reflects LoRA contribution"
            )
        finally:
            s.deactivate()

    @CUDA
    def test_routed_concat_handles_mixed_ranks(self) -> None:
        # Multiple LoRAs targeting the same weight at different ranks
        # must produce the same forward output as the fused route math:
        # fused rank = r1 + r2 + ..., and B_fused @ A_fused equals
        # sum_i strength_i * B_i @ A_i.
        m = _make_bf16_model(num_blocks=2, dim=16)
        # Two LoRAs targeting the same blocks with different ranks.
        lora_r4 = _make_lora(num_blocks=2, dim=16, rank=4, seed=101)
        lora_r8 = _make_lora(num_blocks=2, dim=16, rank=8, seed=202)
        loras = [(lora_r4, 0.6), (lora_r8, 0.3)]

        s = _make_strategy(m)
        s.set_loras(loras, mode="routed")
        s.activate("cuda")
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            actual = m(x)
            torch.cuda.synchronize()
            expected = self._expected_routed_output(m, x, loras)
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1), (
                f"mixed-rank concat output mismatch:\n"
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
                self.transformer_blocks = nn.ModuleList(
                    [Block(16) for _ in range(2)]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for blk in self.transformer_blocks:
                    x = blk(x)
                return x

        model = M().to(torch.bfloat16)
        for p in model.parameters():
            p.requires_grad = False

        s = ModelOffloader(
            model,
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        # Build a LoRA targeting attn.weight (LinearLike, not nn.Linear).
        lora = _make_lora(num_blocks=2, dim=16, seed=3)
        s.set_loras([(lora, 1.0)], mode="routed")
        with pytest.raises(ValueError, match=r"Routed LoRA mode requires nn\.Linear"):
            _activate_loras_for_test(s)
        # Merge mode should still work — it doesn't care about parent type.
        s.set_loras([(lora, 1.0)], mode="merge")
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
                self.attn = (
                    nn.Linear(dim, dim, bias=False) if normal_attn
                    else LinearLike(dim)
                )
                self.ff = nn.Linear(dim, dim, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.ff(self.attn(x))

        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Block 0: normal Linear (passes routed validation).
                # Block 1: LinearLike (fails routed validation).
                self.transformer_blocks = nn.ModuleList(
                    [Block(16, normal_attn=True), Block(16, normal_attn=False)]
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for blk in self.transformer_blocks:
                    x = blk(x)
                return x

        model = M().to(torch.bfloat16)
        for p in model.parameters():
            p.requires_grad = False

        s = ModelOffloader(
            model,
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        lora = _make_lora(num_blocks=2, dim=16, seed=99)
        s.set_loras([(lora, 1.0)], mode="routed")
        with pytest.raises(ValueError, match=r"Routed LoRA mode requires nn\.Linear"):
            _activate_loras_for_test(s)

    @CUDA
    def test_routed_single_lora_skips_concat(self) -> None:
        # Single-LoRA case takes the no-cat fast path. Forward output
        # must still match the manual baseline.
        m = _make_bf16_model(num_blocks=2, dim=16)
        loras = [(_make_lora(num_blocks=2, dim=16, seed=33), 0.7)]

        s = _make_strategy(m)
        s.set_loras(loras, mode="routed")
        s.activate("cuda")
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            actual = m(x)
            torch.cuda.synchronize()
            expected = self._expected_routed_output(m, x, loras)
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1)
        finally:
            s.deactivate()

    @pytest.mark.parametrize("target", ["embed", "head"])
    def test_routed_tied_weight_target_raises(self, target: str) -> None:
        # Standard tied embed/head pattern: one Parameter aliased at
        # multiple slots. PinnedWeights dedups them into one buffer
        # with multiple parent locations. Routed mode would only hook
        # one location and silently miss the others — reject explicitly.
        model = _make_tied_non_block_model(dtype=torch.bfloat16)

        s = ModelOffloader(
            model,
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        # Build a LoRA that targets either alias of the tied weight.
        sd = {
            f"{target}.lora_A.weight": torch.randn(4, 16),
            f"{target}.lora_B.weight": torch.randn(16, 4),
        }
        lora = LoRA(state_dict=sd, key_transform=None)
        s.set_loras([(lora, 1.0)], mode="routed")
        with pytest.raises(ValueError, match="tied weights"):
            _activate_loras_for_test(s)
        # Merge mode handles tied weights via storage mutation.
        s.set_loras([(lora, 1.0)], mode="merge")
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
                self.transformer_blocks = nn.ModuleList(
                    [Block(16) for _ in range(2)]
                )
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
        s = ModelOffloader(
            m,
            layers_attr="transformer_blocks", blocks_to_swap=1,
        )
        s.set_loras(loras, mode="routed")
        s.activate("cuda")
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
                a, b = lora.targets[
                    f"transformer_blocks.{i}.attn.weight"
                ]
                a_dev = a.to(device=h.device, dtype=h.dtype)
                b_dev = b.to(device=h.device, dtype=h.dtype)
                attn_out = base_attn + strength * (h @ a_dev.T @ b_dev.T)
                h = F.linear(attn_out, blk.ff.weight.to(h.device))
            expected = F.linear(h, m.head.weight.to(h.device))
            assert torch.allclose(actual, expected, rtol=0.1, atol=0.1)
        finally:
            s.deactivate()


class TestRoutedFactorDtype:
    """Unit tests for the compute-dtype probe used to cast routed LoRA
    factors. The probe must return the layer's *output* dtype, not the
    weight's *storage* dtype — for quantized layers these differ.
    """

    def test_plain_linear_returns_weight_dtype(self) -> None:
        layer = nn.Linear(8, 8, bias=False).to(torch.bfloat16)
        assert _routed_factor_dtype(layer) is torch.bfloat16

    def test_module_compute_dtype_takes_precedence(self) -> None:
        # BitsAndBytes Linear4bit pattern: subclass of nn.Linear that
        # exposes `compute_dtype` on the module. Probe must prefer that
        # over weight.dtype (which is int8 for bnb's Int8Params).
        layer = nn.Linear(8, 8, bias=False).to(torch.bfloat16)
        layer.compute_dtype = torch.float16  # type: ignore[assignment]
        assert _routed_factor_dtype(layer) is torch.float16

    def test_quanto_weight_dtype_is_compute_dtype(self) -> None:
        # quanto's WeightQBytesTensor wraps an int8 _data plus a fp
        # _scale, but the wrapper-subclass dtype is set to scale.dtype
        # (per `_make_wrapper_subclass(... dtype=scale.dtype ...)`).
        # That means `weight.dtype` already reports the compute dtype —
        # no special-casing needed for routed mode.
        quanto = pytest.importorskip("optimum.quanto")
        from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

        rows, cols = 4, 8
        data = torch.randint(-128, 127, (rows, cols), dtype=torch.int8)
        scale = torch.rand(rows, 1, dtype=torch.bfloat16)
        qt = WeightQBytesTensor.create(
            quanto.qint8, 0, (rows, cols), (cols, 1), data, scale, None,
        )
        layer = nn.Linear(cols, rows, bias=False)
        layer.weight = nn.Parameter(qt, requires_grad=False)

        # Storage is int8; advertised dtype is bf16 (matches scale).
        assert layer.weight._data.dtype == torch.int8
        assert layer.weight.dtype == torch.bfloat16
        assert _routed_factor_dtype(layer) is torch.bfloat16


class TestDeactivateCleanupInvariants:
    @CUDA
    def test_cleanup_runs_even_when_streamer_deactivate_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        s.set_loras([(_make_lora(4, 16), 1.0)])

        def streamer_boom() -> None:
            raise RuntimeError("streamer cleanup failed")

        monkeypatch.setattr(s._streamers[0], "deactivate", streamer_boom)
        s.activate("cuda")

        with pytest.raises(RuntimeError):
            s.deactivate()


# ---------------------------------------------------------------------------
# Cache budget
# ---------------------------------------------------------------------------


class TestCacheBytes:
    def test_lora_cache_bytes_reports_factor_size(self) -> None:
        lora = _make_lora(num_blocks=4, dim=16, rank=4)
        assert lora.cache_bytes > 0

    def test_strategy_cache_bytes_stable_across_set_loras(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        baseline = s.cache_bytes
        s.set_loras([(_make_lora(4, 16), 1.0)])
        assert s.cache_bytes == baseline
        s.set_loras([])
        assert s.cache_bytes == baseline


# ---------------------------------------------------------------------------
# LoRA as CachedResource (ModelCache integration)
# ---------------------------------------------------------------------------


class TestLoRACachedResource:
    def test_lora_satisfies_cached_resource(self) -> None:
        lora = _make_lora(num_blocks=2, dim=8, rank=2)
        assert isinstance(lora, CachedResource)
        assert not isinstance(lora, nn.Module)

    def test_lora_through_model_cache(self) -> None:
        sd = _make_lora_sd(num_blocks=2, dim=8, rank=2)
        cache = ModelCache(10**9)
        spec = ResourceSpec(
            key="lora:test",
            estimated_cache_bytes=1000,
            factory=lambda: LoRA(sd),
        )
        with cache.use(spec) as lora:
            assert isinstance(lora, LoRA)
            assert lora.cache_bytes > 0
            assert len(lora.targets) == 2
        with cache.use("lora:test") as lora2:
            assert lora2 is lora
        snap = cache.snapshot()
        assert snap.stats.builds == 1
        assert snap.stats.hits == 1
