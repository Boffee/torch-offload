"""Streaming a single block list whose blocks are heterogeneously quantized.

The streamer's GPU target pool keys reusable targets by per-block layout
signature, so a block list may mix dtypes / quant formats on the same-named
weights. Before that, the pool was templated off block 0 and a hard layout
check rejected any block that differed — the case these tests cover.
"""

from __future__ import annotations

from tests.conftest import streamed_components

from collections.abc import Sequence

import pytest
import torch
from torch import nn

from torch_offload import ModelOffloaderStore

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _Block(nn.Module):
    """One Linear whose weight dtype/format sets the block's pool layout."""

    def __init__(self, dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast in, compute in the weight's (logical) dtype, cast back, so a
        # block list of mixed precisions composes into one float pipeline.
        return self.lin(x.to(self.lin.weight.dtype)).float()


class _Model(nn.Module):
    def __init__(self, dim: int, dtypes: Sequence[torch.dtype]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim, dt) for dt in dtypes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _frozen_model(dim: int, dtypes: Sequence[torch.dtype]) -> _Model:
    # Heterogeneous-quant weights are frozen in practice; freezing also
    # routes them through the streamer (trainables are not streamed by
    # default) rather than the resident PinnedComponent.
    torch.manual_seed(0)
    model = _Model(dim, dtypes)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _streamed_component(offloader: object):
    components = streamed_components(offloader)  # type: ignore[attr-defined]
    assert len(components) == 1
    return components[0]


def test_heterogeneous_block_list_builds_and_partitions_signatures() -> None:
    """Construction (CPU pinning) no longer rejects mixed-layout blocks, and
    blocks partition into one pool signature per distinct layout."""
    dtypes = [
        torch.float32,
        torch.bfloat16,
        torch.float32,
        torch.bfloat16,
        torch.float32,
    ]
    model = _frozen_model(16, dtypes)

    # Before morphing slots this raised ValueError("Block 1 ... layout
    # differs from block 0 ...") right here.
    store = ModelOffloaderStore.from_module(
        model,
        blocks_attr="blocks",
        num_resident_blocks=1,
        num_prefetch_blocks=2,
    )
    offloader = store.bind(model)

    signatures = _streamed_component(offloader)._block_signatures
    assert len(signatures) == len(dtypes)
    # Two distinct layouts (fp32 vs bf16), alternating with the dtype list.
    assert len(set(signatures)) == 2
    assert signatures[0] == signatures[2] == signatures[4]
    assert signatures[1] == signatures[3]
    assert signatures[0] != signatures[1]


def test_signature_distinguishes_each_dtype() -> None:
    model = _frozen_model(16, [torch.float32, torch.float16, torch.bfloat16])
    offloader = ModelOffloaderStore.from_module(
        model,
        blocks_attr="blocks",
        num_resident_blocks=1,
        num_prefetch_blocks=1,
    ).bind(model)
    signatures = _streamed_component(offloader)._block_signatures
    assert len(set(signatures)) == 3


@CUDA
def test_cuda_streams_mixed_dtype_blocks_matches_reference() -> None:
    dim = 16
    dtypes = [
        torch.float32,
        torch.bfloat16,
        torch.float32,
        torch.bfloat16,
        torch.float32,
        torch.bfloat16,
    ]
    model = _frozen_model(dim, dtypes)
    x = torch.randn(4, dim)
    with torch.no_grad():
        reference = model(x)

    offloader = ModelOffloaderStore.from_module(
        model,
        blocks_attr="blocks",
        num_resident_blocks=1,
        num_prefetch_blocks=2,
    ).bind(model)

    with torch.no_grad(), offloader.use("cuda") as bound:
        streamed = bound(x.cuda()).cpu()

    # Tolerant: ref and streamed share dtypes, so the only delta is
    # CPU-vs-CUDA matmul rounding compounded across the bf16 blocks.
    torch.testing.assert_close(streamed, reference, atol=5e-2, rtol=5e-2)


@CUDA
def test_cuda_morphing_pool_reuses_targets_across_iterations() -> None:
    """A second pass through a mixed block list reuses parked per-signature
    targets — peak residency stays within the configured pool size."""
    dim = 16
    dtypes = [torch.float32, torch.bfloat16] * 4
    model = _frozen_model(dim, dtypes)
    x = torch.randn(2, dim)

    num_resident, num_prefetch = 1, 2
    offloader = ModelOffloaderStore.from_module(
        model,
        blocks_attr="blocks",
        num_resident_blocks=num_resident,
        num_prefetch_blocks=num_prefetch,
    ).bind(model)

    component = _streamed_component(offloader)
    with torch.no_grad(), offloader.use("cuda") as bound:
        bound(x.cuda())
        bound(x.cuda())
        # Concurrency is block-count bounded, independent of how many
        # distinct quant formats interleave.
        assert component.peak_gpu_blocks <= num_resident + num_prefetch


def _int8_quantizer():
    """Return a callable that int8-quantizes a Linear in place, or skip.

    Mirrors ``test_int8_adapter.py``: the int8 adapter targets the
    torchao>=0.17 version-2 ``Int8Tensor`` workflow, so probe whether the
    installed torchao produces a weight this package's registry supports
    and skip (don't fail) when it predates it.
    """
    pytest.importorskip("torchao")
    try:
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
    except ImportError as exc:  # pragma: no cover - torchao API drift
        pytest.skip(f"torchao int8 API unavailable: {exc}")

    from torch_offload.tensor_adapter_registry import select_adapter

    try:
        cfg = Int8WeightOnlyConfig(version=2)
    except TypeError:  # pragma: no cover - older torchao signature
        pytest.skip("installed torchao predates the version-2 Int8Tensor")

    def quantize(linear: nn.Linear) -> None:
        quantize_(linear, cfg)

    probe = nn.Linear(32, 32, bias=False).to(torch.bfloat16)
    quantize(probe)
    try:
        select_adapter(probe.weight.data)
    except NotImplementedError:  # pragma: no cover - torchao format drift
        pytest.skip(
            "installed torchao int8 weight type has no torch-offload adapter"
        )
    return quantize


@CUDA
def test_cuda_streams_mixed_quant_and_plain_blocks() -> None:
    """Realistic case: some blocks torchao-int8 quantized, others plain."""
    quantize = _int8_quantizer()

    dim = 32
    model = _frozen_model(dim, [torch.bfloat16] * 4)
    # Quantize alternate blocks → int8 layout interleaved with plain bf16.
    for block in list(model.blocks)[::2]:
        quantize(block.lin)
    for param in model.parameters():
        param.requires_grad_(False)

    x = torch.randn(4, dim)
    # Reference: the same model run on CUDA without offloading. Move it to
    # the device for the forward, then back to CPU so the offload store
    # pins from host memory as usual.
    model.cuda()
    with torch.no_grad():
        reference = model(x.cuda()).cpu()
    model.cpu()

    offloader = ModelOffloaderStore.from_module(
        model,
        blocks_attr="blocks",
        num_resident_blocks=1,
        num_prefetch_blocks=2,
    ).bind(model)
    assert len(set(_streamed_component(offloader)._block_signatures)) == 2

    with torch.no_grad(), offloader.use("cuda") as bound:
        streamed = bound(x.cuda()).cpu()

    torch.testing.assert_close(streamed, reference, atol=5e-2, rtol=5e-2)
