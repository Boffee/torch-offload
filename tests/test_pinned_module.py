"""Tests for name-based pinned module store/instance primitives."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from torch_offload.pinned_module import (
    PinnedModuleInstance,
    PinnedModuleStore,
)


class TestPinnedModuleStore:
    def test_maps_module_aliases_to_shared_param_backing(self) -> None:
        module = nn.Module()
        shared = nn.Linear(2, 2, bias=False)
        shared.weight.requires_grad_(False)
        module.left = shared
        module.right = shared

        store = PinnedModuleStore.from_module(module)

        assert set(store.params) == {"left.weight", "right.weight"}
        assert store.params["left.weight"] is store.params["right.weight"]
        assert module.left.weight is module.right.weight
        assert module.left.weight.data_ptr() == (
            store.params["left.weight"].make_cpu_param().data_ptr()
        )
        assert store.cache_bytes == store.params["left.weight"].cache_bytes

    def test_storage_alias_mode_shares_distinct_params_with_same_storage(
        self,
    ) -> None:
        module = nn.Module()
        shared = torch.randn(2, 2)
        module.a = nn.Parameter(shared, requires_grad=False)
        module.b = nn.Parameter(shared, requires_grad=False)

        store = PinnedModuleStore.from_module(module)

        assert set(store.params) == {"a", "b"}
        assert store.params["a"] is store.params["b"]
        assert module.a is module.b
        assert module.a.data_ptr() == store.params["a"].make_cpu_param().data_ptr()

    def test_object_alias_mode_keeps_distinct_param_objects_separate(
        self,
    ) -> None:
        module = nn.Module()
        shared = torch.randn(2, 2)
        module.a = nn.Parameter(shared, requires_grad=False)
        module.b = nn.Parameter(shared, requires_grad=False)

        store = PinnedModuleStore.from_module(
            module,
            param_alias_mode="object",
        )

        assert store.params["a"] is not store.params["b"]
        assert module.a is not module.b
        assert module.a.data_ptr() != module.b.data_ptr()

    def test_keeps_distinct_param_backings_separate(self) -> None:
        module = nn.Module()
        module.left = nn.Linear(2, 2, bias=False)
        module.right = nn.Linear(2, 2, bias=False)
        module.left.weight.requires_grad_(False)
        module.right.weight.requires_grad_(False)

        store = PinnedModuleStore.from_module(module)

        assert set(store.params) == {"left.weight", "right.weight"}
        assert store.params["left.weight"] is not store.params["right.weight"]
        assert store.cache_bytes == (
            store.params["left.weight"].cache_bytes
            + store.params["right.weight"].cache_bytes
        )

    def test_maps_buffer_aliases_to_shared_backing(self) -> None:
        module = nn.Module()
        shared = torch.tensor([1.0, 2.0])
        module.register_buffer("running", shared)
        module.register_buffer("running_alias", shared)

        store = PinnedModuleStore.from_module(module)

        assert store.params == {}
        assert set(store.buffers) == {"running", "running_alias"}
        assert store.buffers["running"] is store.buffers["running_alias"]
        assert module.running is module.running_alias
        assert module.running.data_ptr() == store.buffers["running"].tensor.data_ptr()
        assert store.cache_bytes == store.buffers["running"].cache_bytes

    def test_can_exclude_buffers(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        module.register_buffer("running", torch.randn(2))

        store = PinnedModuleStore.from_module(module, include_buffers=False)

        assert set(store.params) == {"weight"}
        assert store.buffers == {}
        assert store.cache_bytes == store.params["weight"].cache_bytes

    def test_runs_param_validation_before_pinning(self) -> None:
        module = nn.Module()
        original = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        module.weight = original

        def reject(name: str, param: nn.Parameter) -> None:
            assert name == "weight"
            assert param is original
            raise ValueError("rejected")

        with pytest.raises(ValueError, match="rejected"):
            PinnedModuleStore.from_module(module, validate_param=reject)

        assert module.weight is original


class TestPinnedModuleInstance:
    def test_binds_same_store_to_multiple_modules(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        prototype.register_buffer("running", torch.randn(2))
        store = PinnedModuleStore.from_module(prototype)

        first = PinnedModuleInstance.from_store(store, prototype)

        second_module = nn.Module()
        second_module.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        second_module.register_buffer("running", torch.randn(2))
        second = PinnedModuleInstance.from_store(store, second_module)

        pinned = store.params["weight"]
        first_cpu = first.cpu_params_by_pinned_id[id(pinned)]
        second_cpu = second.cpu_params_by_pinned_id[id(pinned)]

        assert first.cache_bytes == store.cache_bytes
        assert first_cpu is not second_cpu
        assert first_cpu.data_ptr() == pinned.make_cpu_param().data_ptr()
        assert second_cpu.data_ptr() == pinned.make_cpu_param().data_ptr()
        assert prototype.weight is first_cpu
        assert second_module.weight is second_cpu
        assert prototype.running is store.buffers["running"].tensor
        assert second_module.running is store.buffers["running"].tensor

    def test_does_not_store_slots_or_parent_leaf_state(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = PinnedModuleStore.from_module(module)

        instance = PinnedModuleInstance.from_store(store, module)

        assert not hasattr(instance, "param_slots")
        assert not hasattr(instance, "buffer_slots")
        assert not hasattr(instance, "parent")
        assert not hasattr(instance, "leaf")

    def test_restores_tied_params_with_one_cpu_wrapper(self) -> None:
        prototype = nn.Module()
        shared = nn.Linear(2, 2, bias=False)
        shared.weight.requires_grad_(False)
        prototype.left = shared
        prototype.right = shared
        store = PinnedModuleStore.from_module(prototype)

        target = nn.Module()
        target_shared = nn.Linear(2, 2, bias=False)
        target_shared.weight.requires_grad_(False)
        target.left = target_shared
        target.right = target_shared
        instance = PinnedModuleInstance.from_store(store, target)

        pinned = store.params["left.weight"]
        cpu_param = instance.cpu_params_by_pinned_id[id(pinned)]

        assert store.params["left.weight"] is store.params["right.weight"]
        assert target.left.weight is cpu_param
        assert target.right.weight is cpu_param

    def test_trainable_restore_preserves_parameter_wrapper(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=True)
        store = PinnedModuleStore.from_module(prototype)

        target = nn.Module()
        target.weight = nn.Parameter(torch.randn(2, 2), requires_grad=True)
        target_param = target.weight

        instance = PinnedModuleInstance.from_store(store, target)
        cpu_param = instance.cpu_params_by_pinned_id[id(store.params["weight"])]

        assert target.weight is target_param
        assert target.weight.data_ptr() == cpu_param.data_ptr()

    def test_preserves_target_buffer_persistence(self) -> None:
        prototype = nn.Module()
        prototype.register_buffer("running", torch.randn(2), persistent=True)
        store = PinnedModuleStore.from_module(prototype)

        target = nn.Module()
        target.register_buffer("running", torch.randn(2), persistent=False)

        PinnedModuleInstance.from_store(store, target)

        assert target.running is store.buffers["running"].tensor
        assert "running" in target._non_persistent_buffers_set

    def test_rejects_missing_param_name(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = PinnedModuleStore.from_module(prototype)

        with pytest.raises(ValueError, match="missing pinned param names"):
            PinnedModuleInstance.from_store(store, nn.Module())

    def test_rejects_missing_buffer_name(self) -> None:
        prototype = nn.Module()
        prototype.register_buffer("running", torch.randn(2))
        store = PinnedModuleStore.from_module(prototype)

        with pytest.raises(ValueError, match="missing pinned buffer names"):
            PinnedModuleInstance.from_store(store, nn.Module())

    def test_rejects_requires_grad_mismatch(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = PinnedModuleStore.from_module(prototype)

        target = nn.Module()
        target.weight = nn.Parameter(torch.randn(2, 2), requires_grad=True)

        with pytest.raises(ValueError, match="requires_grad mismatch"):
            PinnedModuleInstance.from_store(store, target)

    def test_rejects_param_layout_mismatch(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = PinnedModuleStore.from_module(prototype)

        target = nn.Module()
        target.weight = nn.Parameter(torch.randn(3, 2), requires_grad=False)

        with pytest.raises(ValueError, match="Param 'weight' layout mismatch"):
            PinnedModuleInstance.from_store(store, target)

    def test_rejects_buffer_layout_mismatch(self) -> None:
        prototype = nn.Module()
        prototype.register_buffer("running", torch.randn(2))
        store = PinnedModuleStore.from_module(prototype)

        target = nn.Module()
        target.register_buffer("running", torch.randn(3))

        with pytest.raises(ValueError, match="Buffer 'running' layout mismatch"):
            PinnedModuleInstance.from_store(store, target)

    def test_rejects_param_alias_topology_mismatch(self) -> None:
        prototype = nn.Module()
        shared = nn.Linear(2, 2, bias=False)
        shared.weight.requires_grad_(False)
        prototype.left = shared
        prototype.right = shared
        store = PinnedModuleStore.from_module(prototype)

        target = nn.Module()
        target.left = nn.Linear(2, 2, bias=False)
        target.right = nn.Linear(2, 2, bias=False)
        target.left.weight.requires_grad_(False)
        target.right.weight.requires_grad_(False)

        with pytest.raises(ValueError, match="param alias topology mismatch"):
            PinnedModuleInstance.from_store(store, target)

    def test_rejects_untracked_param_alias(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = PinnedModuleStore.from_module(prototype)

        target = nn.Module()
        shared = torch.randn(2, 2)
        target.weight = nn.Parameter(shared, requires_grad=False)
        target.extra = nn.Parameter(shared, requires_grad=False)

        with pytest.raises(ValueError, match="param alias topology mismatch"):
            PinnedModuleInstance.from_store(store, target)

    def test_rejects_buffer_alias_topology_mismatch(self) -> None:
        prototype = nn.Module()
        shared = torch.randn(2)
        prototype.register_buffer("running", shared)
        prototype.register_buffer("running_alias", shared)
        store = PinnedModuleStore.from_module(prototype)

        target = nn.Module()
        target.register_buffer("running", torch.randn(2))
        target.register_buffer("running_alias", torch.randn(2))

        with pytest.raises(ValueError, match="buffer alias topology mismatch"):
            PinnedModuleInstance.from_store(store, target)
