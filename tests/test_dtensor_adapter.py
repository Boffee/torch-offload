"""Tests for the movement-only :class:`DTensorAdapter` (tensor-parallel weights).

A ``DTensor`` weight needs a ``DeviceMesh`` and a process group, so the whole
module requires CUDA and initializes a single-rank group. Single-rank means
``local == global`` (the N≥2 sharding traps — ``data_ptr()==0`` dedup collapse
and ``cache_bytes`` over-accounting — can't be reproduced here), so those are
covered by reasoning: the adapter keys identity/bytes off the *local* shard.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch
from torch import nn

from torch_offload import ModelOffloaderStore
from torch_offload.dtensor_adapter import DTensorAdapter
from torch_offload.pinned_param import PinnedParam
from torch_offload.tensor_adapters import (
    CpuRoundTripTensorAdapter,
    DequantRequantTensorAdapter,
    ParameterDataSwapTensorAdapter,
    RegularAdapter,
    TensorCopyIntoAdapter,
)
from torch_offload.tensor_adapter_registry import select_adapter, tensor_id

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DTensor tensor-parallel weights need a CUDA mesh + process group",
)


@pytest.fixture(scope="module")
def tp_mesh() -> Any:
    import torch.distributed as dist
    from torch.distributed.tensor import init_device_mesh

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29593")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    created = False
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)
        created = True
    torch.cuda.set_device(0)
    mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("tp",))
    yield mesh
    if created and dist.is_initialized():
        dist.destroy_process_group()


def _shard(dim: int = 0) -> Any:
    from torch.distributed.tensor import Shard

    return Shard(dim)


def _dtensor_weight(
    mesh: Any,
    *,
    rows: int = 16,
    cols: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    placement: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch.distributed.tensor import distribute_tensor

    full = torch.randn(rows, cols, dtype=dtype, device="cuda")
    dt = distribute_tensor(full, mesh, [placement or _shard(0)])
    return dt, full


def _is_dtensor(t: torch.Tensor) -> bool:
    from torch.distributed.tensor import DTensor

    return isinstance(t, DTensor)


class TestDTensorAdapter:
    def test_matches_and_dispatches(self, tp_mesh: Any) -> None:
        dt, _ = _dtensor_weight(tp_mesh)
        assert DTensorAdapter.matches(dt)
        assert isinstance(select_adapter(dt), DTensorAdapter)
        assert not DTensorAdapter.matches(torch.zeros(4, 4))

    def test_delegates_local_shard_to_inner_adapter(self, tp_mesh: Any) -> None:
        dt, _ = _dtensor_weight(tp_mesh)
        pinned_param = PinnedParam(nn.Parameter(dt, requires_grad=False))
        # A plain local shard is moved by the registry's RegularAdapter — the
        # DTensorAdapter only adds the distributed wrapper on top.
        assert isinstance(pinned_param.pinned_state.inner, RegularAdapter)

    def test_pinned_param_roundtrip_reconstructs_dtensor(self, tp_mesh: Any) -> None:
        dt, full = _dtensor_weight(tp_mesh)
        pinned_param = PinnedParam(nn.Parameter(dt, requires_grad=False))

        # Resting state: still a DTensor (type-stable), but on a CPU mesh so
        # the local shard stays on the host (no GPU memory held).
        cpu = pinned_param.make_cpu_param()
        assert _is_dtensor(cpu.data)
        assert cpu.data.device_mesh.device_type == "cpu"
        assert cpu.data.to_local().device.type == "cpu"
        assert cpu.data.placements == dt.placements
        assert torch.equal(cpu.data.to_local(), dt.to_local().cpu())

        # Resident state: the DTensor is reconstructed on the GPU.
        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()

        assert _is_dtensor(gpu_param.data)
        assert gpu_param.data.to_local().is_cuda
        assert gpu_param.data.placements == dt.placements
        assert gpu_param.data.device_mesh == dt.device_mesh
        assert torch.equal(gpu_param.data.full_tensor(), full)

    def test_gpu_param_aliases_inner_storage_for_refill(self, tp_mesh: Any) -> None:
        # The pooled streaming path reuses one wrapper across loads, refilling
        # its buffers in place. from_local must alias the inner GPU storage
        # (not copy) so refills are visible through the wrapper.
        dt, _ = _dtensor_weight(tp_mesh)
        pinned_param = PinnedParam(nn.Parameter(dt, requires_grad=False))
        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()

        assert gpu_param.data.to_local().data_ptr() == gpu_state.inner_gpu.data.data_ptr()

    def test_tensor_id_and_layout_track_placements(self, tp_mesh: Any) -> None:
        from torch.distributed.tensor import Replicate, distribute_tensor

        full = torch.randn(16, 8, dtype=torch.bfloat16, device="cuda")
        sharded = distribute_tensor(full, tp_mesh, [_shard(0)])
        replicated = distribute_tensor(full, tp_mesh, [Replicate()])

        assert DTensorAdapter.layout_signature(sharded) != DTensorAdapter.layout_signature(replicated)
        assert tensor_id(sharded) != tensor_id(replicated)
        assert tensor_id(sharded)[0] == "dtensor"

    def test_bind_layout_relaxes_inner_dtype_keeps_global_shape(
        self, tp_mesh: Any
    ) -> None:
        bf16 = _dtensor_weight(tp_mesh, dtype=torch.bfloat16)[0]
        fp32 = _dtensor_weight(tp_mesh, dtype=torch.float32)[0]

        # bind_layout delegates to the inner adapter's bind signature, which
        # drops dtype (for meta-skeleton binding) — so the two compare equal;
        # the strict layout_signature keeps dtype, so they differ.
        assert DTensorAdapter.layout_signature(bf16) != DTensorAdapter.layout_signature(fp32)
        assert DTensorAdapter.bind_layout_signature(
            bf16
        ) == DTensorAdapter.bind_layout_signature(fp32)
        # Both keys carry the GLOBAL shape (gpu_param replays it; uneven shards
        # are not pinned by the local shape alone).
        assert tuple(bf16.shape) in DTensorAdapter.layout_signature(bf16)
        assert tuple(bf16.shape) in DTensorAdapter.bind_layout_signature(bf16)

    def test_tensor_id_keys_off_local_not_dtensor_dataptr(self, tp_mesh: Any) -> None:
        dt, _ = _dtensor_weight(tp_mesh)
        # The DTensor's own data_ptr is 0 — using it would collapse tied-weight
        # dedup. Two independently-allocated DTensors must get distinct ids.
        assert dt.data_ptr() == 0
        other, _ = _dtensor_weight(tp_mesh)
        assert tensor_id(dt) != tensor_id(other)
        assert tensor_id(dt) == tensor_id(dt)

    def test_cache_bytes_counts_local_shard(self, tp_mesh: Any) -> None:
        dt, _ = _dtensor_weight(tp_mesh, rows=16, cols=8)
        pinned_param = PinnedParam(nn.Parameter(dt, requires_grad=False))
        local = dt.to_local()
        # Local-shard bytes (== global only because world_size==1; on N ranks
        # this is the ~1/N local footprint, not the global numel).
        assert DTensorAdapter.cache_bytes(pinned_param.pinned_state) == (local.numel() * local.element_size())

    def test_compute_dtype_delegates_to_local(self, tp_mesh: Any) -> None:
        dt, _ = _dtensor_weight(tp_mesh, dtype=torch.bfloat16)
        assert DTensorAdapter.compute_dtype(dt) is torch.bfloat16

    def test_compute_dtype_via_pinned_param_property(self, tp_mesh: Any) -> None:
        # Regression: PinnedParam.compute_dtype feeds the bare local shard
        # (what cpu_param yields, not a DTensor) into the adapter; it must not
        # require a live DTensor. Every other adapter asserts this property.
        dt, _ = _dtensor_weight(tp_mesh, dtype=torch.bfloat16)
        pinned_param = PinnedParam(nn.Parameter(dt, requires_grad=False))
        assert pinned_param.compute_dtype is torch.bfloat16

    def test_streamed_offloader_reconstructs_dtensor_blocks(
        self, tp_mesh: Any
    ) -> None:
        # The real production path: bind a ModelOffloader over DTensor-weighted
        # blocks, activate (gpu_param rebuilds the DTensor), deactivate
        # (cpu_param restores the bare local shard).
        class Block(nn.Module):
            def __init__(self, w: torch.Tensor) -> None:
                super().__init__()
                self.weight = nn.Parameter(w, requires_grad=False)

        class Net(nn.Module):
            def __init__(self, blocks: list[nn.Module]) -> None:
                super().__init__()
                self.blocks = nn.ModuleList(blocks)

        net = Net(
            [Block(_dtensor_weight(tp_mesh)[0]), Block(_dtensor_weight(tp_mesh)[0])]
        )
        store = ModelOffloaderStore.from_module(
            net, blocks_attr="blocks", num_resident_blocks=2, num_prefetch_blocks=0
        )
        pw = store.bind(net, skip_checkpointing_check=True)
        try:
            # resting: a DTensor on a CPU mesh (local on the host)
            resting = net.blocks[0].weight.data
            assert _is_dtensor(resting)
            assert resting.device_mesh.device_type == "cpu"
            with pw.use("cuda"):
                for blk in net.blocks:
                    assert _is_dtensor(blk.weight.data)
                    assert blk.weight.data.device_mesh.device_type == "cuda"
                    assert blk.weight.data.to_local().is_cuda
            # back to resting (CPU mesh) after deactivate
            assert net.blocks[0].weight.data.device_mesh.device_type == "cpu"
        finally:
            pw.deactivate()

    def test_advertises_movement_only(self, tp_mesh: Any) -> None:
        # Frozen-inference scope: no CPU round-trip, no dequant/requant, no
        # copy_into, no trainable .data swap — even if the inner adapter has
        # them (routed LoRA, not merged, is the inference path).
        adapter = select_adapter(_dtensor_weight(tp_mesh)[0])
        assert not isinstance(adapter, CpuRoundTripTensorAdapter)
        assert not isinstance(adapter, DequantRequantTensorAdapter)
        assert not isinstance(adapter, TensorCopyIntoAdapter)
        assert not isinstance(adapter, ParameterDataSwapTensorAdapter)

    def test_composes_with_quantized_local_shard(self, tp_mesh: Any) -> None:
        # The crown-jewel claim: one adapter composes with every quant adapter.
        # A DTensor wrapping a TorchAO Float8Tensor must reuse Float8Adapter
        # for the local shard with no DTensor-specific quant code.
        #
        # Build via from_local (wrap a per-rank shard) — the canonical TP
        # construction path frameworks use. distribute_tensor (split a full
        # tensor) hits a TorchAO Float8Tensor torch.chunk dispatch bug; the
        # adapter is agnostic to how the DTensor was created.
        pytest.importorskip("torchao")
        from torch.distributed.tensor import DTensor

        from torch_offload.float8_adapter import Float8Adapter

        try:
            from torchao.quantization import (
                Float8WeightOnlyConfig,
                quantize_,
            )

            layer = nn.Linear(8, 16, bias=False).to(torch.bfloat16).cuda()
            quantize_(layer, Float8WeightOnlyConfig())
            f8 = layer.weight.data  # a Float8Tensor
            dt = DTensor.from_local(f8, tp_mesh, [_shard(0)], run_check=False)
        except Exception as exc:  # env/version dependent
            pytest.skip(f"torchao Float8 DTensor unavailable: {exc}")

        assert isinstance(select_adapter(dt), DTensorAdapter)
        assert isinstance(select_adapter(dt.to_local()), Float8Adapter)

        pinned_param = PinnedParam(nn.Parameter(dt, requires_grad=False))
        assert isinstance(pinned_param.pinned_state.inner, Float8Adapter)

        gpu_state = pinned_param.allocate_gpu_storage(torch.device("cuda"))
        pinned_param.copy_to_gpu(gpu_state, non_blocking=True)
        gpu_param = pinned_param.make_gpu_param(gpu_state)
        torch.cuda.synchronize()

        assert _is_dtensor(gpu_param.data)
        # the reconstructed local shard is still the Float8 quant subclass
        assert isinstance(select_adapter(gpu_param.data.to_local()), Float8Adapter)
        assert gpu_param.data.placements == dt.placements
