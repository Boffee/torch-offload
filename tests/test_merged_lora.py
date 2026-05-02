"""Tests for LoRA merge via ``ModelOffloader.set_loras()``.

Covers LoRA construction validation, set_loras matching, lifecycle
(activate/deactivate), LoRA switching, and forward-output correctness
against a manually-merged baseline.

Most lifecycle tests run on CPU (the merge math is device-agnostic);
CUDA-only tests gate on availability.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torch_offload import (
    LoRA,
    ModelCache,
    ModelOffloader,
    ResourceSpec,
)
from torch_offload.lora import KeyTransformT
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
        out = out + strength * (b.to(base.dtype) @ a.to(base.dtype))
    return out


def _make_strategy(
    model: nn.Module, device: str = "cpu", blocks_to_swap: int = 1,
) -> ModelOffloader:
    """Shorthand for constructing the strategy with sensible defaults."""
    return ModelOffloader(
        model, torch.device(device),
        layers_attr="transformer_blocks",
        blocks_to_swap=blocks_to_swap,
    )


def _has_transform(strategy: ModelOffloader, target_key: str) -> bool:
    """Check whether a transform is attached for the given target."""
    buf = strategy._reverse_index.get(target_key)
    return buf is not None and buf.transform is not None


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
# set_loras validation
# ---------------------------------------------------------------------------


class TestSetLorasValidation:
    def test_rejects_target_shape_mismatch(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {
            "transformer_blocks.0.attn.lora_A.weight": torch.randn(4, 16),
            "transformer_blocks.0.attn.lora_B.weight": torch.randn(8, 4),
        }
        with pytest.raises(ValueError, match="shape mismatch"):
            s.set_loras([(LoRA(state_dict=sd), 1.0)])

    def test_accepts_fp32_lora_target(self) -> None:
        m = _make_bf16_model().to(torch.float32)
        for p in m.parameters():
            p.requires_grad = False
        s = _make_strategy(m)
        s.set_loras([(_make_lora(4, 16), 1.0)])
        assert _has_transform(s, "transformer_blocks.0.attn.weight")

    def test_non_block_targets_matched(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        sd = {
            "embed.lora_A.weight": torch.randn(4, 16),
            "embed.lora_B.weight": torch.randn(16, 4),
        }
        s.set_loras([(LoRA(state_dict=sd), 1.0)])
        assert _has_transform(s, "embed.weight")

    def test_key_transform_strips_prefix(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16, prefix="diffusion_model.")
        s.set_loras([(lora, 1.0)])
        assert _has_transform(s, "transformer_blocks.0.attn.weight")

    def test_key_transform_none_matches_exact_keys(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16, key_transform=None)
        s.set_loras([(lora, 1.0)])
        assert _has_transform(s, "transformer_blocks.0.attn.weight")

    def test_key_transform_none_skips_prefixed_keys(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        lora = _make_lora(4, 16, prefix="diffusion_model.", key_transform=None)
        s.set_loras([(lora, 1.0)])
        assert not _has_transform(s, "transformer_blocks.0.attn.weight")

    @CUDA
    def test_set_loras_raises_while_active(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m, device="cuda")
        s.set_loras([(_make_lora(4, 16), 1.0)])
        s.activate()
        try:
            with pytest.raises(RuntimeError, match="inactive"):
                s.set_loras([])
        finally:
            s.deactivate()

    def test_set_loras_clears_previous(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m)
        s.set_loras([(_make_lora(4, 16, rank=4), 1.0)])
        assert _has_transform(s, "transformer_blocks.0.attn.weight")
        s.set_loras([])
        assert not _has_transform(s, "transformer_blocks.0.attn.weight")

    def test_accepts_fp16_base(self) -> None:
        m = _make_bf16_model().to(torch.float16)
        for p in m.parameters():
            p.requires_grad = False
        s = _make_strategy(m)
        assert s.cache_bytes > 0


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    @CUDA
    def test_activate_runs_components(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m, device="cuda")
        s.set_loras([(_make_lora(4, 16), 1.0)])
        try:
            s.activate()
            assert m.embed.weight.is_cuda
            assert m.head.weight.is_cuda
        finally:
            s.deactivate()

    @CUDA
    def test_deactivate_returns_to_pinned(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m, device="cuda")
        s.set_loras([(_make_lora(4, 16), 1.0)])
        s.activate()
        s.deactivate()
        assert m.embed.weight.is_pinned()
        assert m.head.weight.is_pinned()

    @CUDA
    def test_reactivation_with_different_loras(self) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m, device="cuda")
        s.set_loras([(_make_lora(4, 16, seed=1), 1.0)])
        s.activate()
        s.deactivate()
        s.set_loras([(_make_lora(4, 16, seed=2), 1.0)])
        s.activate()
        s.deactivate()
        assert m.embed.weight.is_pinned()

    @CUDA
    def test_activate_with_no_loras_runs_base_only(self) -> None:
        m = _make_bf16_model()
        captured = m.transformer_blocks[0].attn.weight.detach().clone()
        s = _make_strategy(m, device="cuda")
        s.activate()
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            for blk in m.transformer_blocks:
                x = blk(x)
            torch.cuda.synchronize()
            actual = m.transformer_blocks[0].attn.weight.detach()
            assert torch.allclose(
                actual, captured.to("cuda"), rtol=0.0, atol=0.0,
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
        s = _make_strategy(m, device="cuda")
        s.set_loras(loras)
        s.activate()
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            for blk in m.transformer_blocks:
                x = blk(x)
            torch.cuda.synchronize()
            for i in range(4):
                expected = _expected_merged_weight(
                    captured_base[i], loras, i, "attn.weight",
                ).to("cuda")
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
        s = _make_strategy(m, device="cuda")
        s.set_loras([(lora, 0.7)])
        s.activate()
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            for blk in m.transformer_blocks:
                x = blk(x)
            torch.cuda.synchronize()
            for i in range(4):
                expected = _expected_merged_weight(
                    captured_base[i], [(lora, 0.7)], i, "attn.weight",
                ).to("cuda")
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
        s = _make_strategy(m, device="cuda")
        s.set_loras([(lora, 0.5)])
        s.activate()
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
        s = _make_strategy(m, device="cuda")
        s.set_loras([(lora, 0.7)])
        s.activate()
        try:
            x = torch.randn(2, 16, dtype=torch.bfloat16, device="cuda")
            m(x)
            torch.cuda.synchronize()
            for i in range(4):
                expected = _expected_merged_weight(
                    captured_base[i], [(lora, 0.7)], i, "attn.weight",
                ).to("cuda")
                actual = m.transformer_blocks[i].attn.base_layer.weight.detach()
                assert torch.allclose(actual, expected, rtol=0.01, atol=0.01), (
                    f"block {i} PEFT-wrapped merge mismatch"
                )
        finally:
            s.deactivate()


# ---------------------------------------------------------------------------
# Cleanup invariants
# ---------------------------------------------------------------------------


class TestDeactivateCleanupInvariants:
    @CUDA
    def test_cleanup_runs_even_when_streamer_deactivate_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        m = _make_bf16_model()
        s = _make_strategy(m, device="cuda")
        s.set_loras([(_make_lora(4, 16), 1.0)])

        def streamer_boom() -> None:
            raise RuntimeError("streamer cleanup failed")

        monkeypatch.setattr(s._streamers[0], "deactivate", streamer_boom)
        s.activate()

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
