"""Tests for name-based pinned module store/instance primitives."""

from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

from torch_offload import pinned_module
from torch_offload.pinned_buffer import PinnedBuffer
from torch_offload.pinned_module import (
    PinnedBufferTarget,
    PinnedModuleInstance,
    PinnedModuleStore,
    PinnedModuleTarget,
    PinnedParamTarget,
)
from torch_offload.pinned_param import PinnedParam


class _FakePinnedParam:
    def __init__(
        self,
        target_data: torch.Tensor,
        *,
        requires_grad: bool = False,
    ) -> None:
        self.allocated = 0
        self.copied = 0
        self.copied_back = 0
        self.validated = 0
        self.copy_to_cpu_non_blocking: list[bool] = []
        self.requires_grad = requires_grad
        self.target_data = target_data
        self.target_layout = PinnedParam.target_layout_for(
            nn.Parameter(target_data, requires_grad=requires_grad),
        )

    @property
    def cache_bytes(self) -> int:
        return self.target_data.numel() * self.target_data.element_size()

    def make_cpu_param(self) -> nn.Parameter:
        return nn.Parameter(
            torch.empty_like(self.target_data),
            requires_grad=self.requires_grad,
        )

    def allocate_gpu_storage(self, device: torch.device) -> object:
        self.allocated += 1
        return {"device": device}

    def make_gpu_param(self, target_state: object) -> nn.Parameter:
        del target_state
        return nn.Parameter(
            self.target_data.clone(),
            requires_grad=self.requires_grad,
        )

    def copy_to_gpu(
        self,
        target_state: object,
        *,
        non_blocking: bool = False,
    ) -> None:
        del target_state, non_blocking
        self.copied += 1

    def copy_to_cpu(
        self,
        target_state: object,
        *,
        non_blocking: bool = False,
    ) -> None:
        del target_state
        self.copied_back += 1
        self.copy_to_cpu_non_blocking.append(non_blocking)

    def validate_parameter_data_swap_target(self) -> None:
        self.validated += 1


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

    def test_shares_distinct_params_with_same_storage(
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

    def test_rejects_storage_aliases_with_mixed_requires_grad(self) -> None:
        module = nn.Module()
        shared = torch.randn(2, 2)
        module.frozen = nn.Parameter(shared, requires_grad=False)
        module.trainable = nn.Parameter(shared, requires_grad=True)

        with pytest.raises(ValueError, match="mixed requires_grad"):
            PinnedModuleStore.from_module(module)

        assert module.frozen.requires_grad is False
        assert module.trainable.requires_grad is True

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

        store = PinnedModuleStore.from_module(module, include_buffer_names=set())

        assert set(store.params) == {"weight"}
        assert store.buffers == {}
        assert store.cache_bytes == store.params["weight"].cache_bytes

    def test_can_include_params_by_name(self) -> None:
        module = nn.Module()
        module.keep = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        module.skip = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        skipped_param = module.skip

        store = PinnedModuleStore.from_module(
            module,
            include_param_names={"keep"},
        )

        assert set(store.params) == {"keep"}
        assert module.keep.data_ptr() == store.params["keep"].make_cpu_param().data_ptr()
        assert module.skip is skipped_param

    def test_param_include_names_can_split_shared_storage(self) -> None:
        module = nn.Module()
        shared = torch.randn(2, 2)
        module.keep = nn.Parameter(shared, requires_grad=False)
        module.skip = nn.Parameter(shared, requires_grad=False)
        skipped_param = module.skip

        store = PinnedModuleStore.from_module(
            module,
            include_param_names={"keep"},
        )

        assert set(store.params) == {"keep"}
        assert module.keep.data_ptr() == store.params["keep"].make_cpu_param().data_ptr()
        assert module.skip is skipped_param

    def test_empty_include_name_sets_pin_nothing(self) -> None:
        module = nn.Module()
        param = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        buffer = torch.randn(2)
        module.weight = param
        module.register_buffer("running", buffer)

        store = PinnedModuleStore.from_module(
            module,
            include_param_names=set(),
            include_buffer_names=set(),
        )

        assert store.params == {}
        assert store.buffers == {}
        assert module.weight is param
        assert module.running is buffer

    def test_rejects_unknown_param_include_names(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)

        with pytest.raises(ValueError, match="unknown names: 'missing'"):
            PinnedModuleStore.from_module(
                module,
                include_param_names={"missing"},
            )

    def test_can_include_buffers_by_name(self) -> None:
        module = nn.Module()
        skipped_buffer = torch.randn(2)
        module.register_buffer("keep", torch.randn(2))
        module.register_buffer("skip", skipped_buffer)

        store = PinnedModuleStore.from_module(
            module,
            include_buffer_names={"keep"},
        )

        assert store.params == {}
        assert set(store.buffers) == {"keep"}
        assert module.keep is store.buffers["keep"].tensor
        assert module.skip is skipped_buffer

    def test_buffer_include_names_can_split_shared_storage(self) -> None:
        module = nn.Module()
        shared = torch.randn(2)
        module.register_buffer("running", shared)
        module.register_buffer("running_alias", shared)

        store = PinnedModuleStore.from_module(
            module,
            include_buffer_names={"running"},
        )

        assert set(store.buffers) == {"running"}
        assert module.running is store.buffers["running"].tensor
        assert module.running_alias is shared

    def test_rejects_unknown_buffer_include_names(self) -> None:
        module = nn.Module()
        module.register_buffer("running", torch.randn(2))

        with pytest.raises(ValueError, match="unknown names: 'missing'"):
            PinnedModuleStore.from_module(
                module,
                include_buffer_names={"missing"},
            )

    def test_from_module_validates_trainable_swap_before_restore(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.zeros(2), requires_grad=True)
        calls: list[str] = []

        def fail_validate(_pinned: PinnedParam) -> None:
            calls.append("validate")
            raise NotImplementedError("unsupported")

        def fail_restore(*_args: object, **_kwargs: object) -> None:
            calls.append("restore")
            raise AssertionError("restore should not run before validation")

        monkeypatch.setattr(
            PinnedParam,
            "validate_parameter_data_swap_target",
            fail_validate,
        )
        monkeypatch.setattr(pinned_module, "_restore_params", fail_restore)

        with pytest.raises(NotImplementedError, match="Trainable param 'weight'"):
            PinnedModuleStore.from_module(module)

        assert calls == ["validate"]


class TestPinnedModuleInstance:
    def test_allocate_target_dedupes_alias_backings(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_empty_like = torch.empty_like

        def fake_empty_like(
            tensor: torch.Tensor,
            *,
            device: torch.device | None = None,
        ) -> torch.Tensor:
            if device is not None:
                assert device == torch.device("cuda")
            return original_empty_like(tensor)

        monkeypatch.setattr(torch, "empty_like", fake_empty_like)
        pinned_param = _FakePinnedParam(torch.empty(2, 2))
        pinned_buffer = PinnedBuffer.clone(torch.randn(2))
        store = PinnedModuleStore(
            params={
                "left.weight": cast(PinnedParam, pinned_param),
                "right.weight": cast(PinnedParam, pinned_param),
            },
            buffers={
                "running": pinned_buffer,
                "running_alias": pinned_buffer,
            },
        )
        instance = PinnedModuleInstance(
            module=nn.Module(),
            params=store.params,
            buffers=store.buffers,
            cpu_params_by_pinned_id={
                id(pinned_param): pinned_param.make_cpu_param(),
            },
        )

        target = instance.allocate_target(torch.device("cuda"))

        assert pinned_param.allocated == 1
        assert target.param_targets["left.weight"] is target.param_targets["right.weight"]
        assert target.buffer_targets["running"] is target.buffer_targets["running_alias"]

    def test_allocate_target_rejects_non_cuda_device(self) -> None:
        instance = PinnedModuleInstance(
            module=nn.Module(),
            params={},
            buffers={},
            cpu_params_by_pinned_id={},
        )

        with pytest.raises(ValueError, match="requires a CUDA device"):
            instance.allocate_target(torch.device("cpu"))

    def test_load_to_target_copies_and_hooks_once_for_aliases(self) -> None:
        module = nn.Module()
        shared = nn.Linear(2, 2, bias=False)
        shared.weight.requires_grad_(False)
        module.left = shared
        module.right = shared
        pinned = _FakePinnedParam(torch.ones(2, 2))
        store = PinnedModuleStore(
            params={
                "left.weight": cast(PinnedParam, pinned),
                "right.weight": cast(PinnedParam, pinned),
            },
            buffers={},
        )
        instance = PinnedModuleInstance(
            module=module,
            params=store.params,
            buffers=store.buffers,
            cpu_params_by_pinned_id={id(pinned): pinned.make_cpu_param()},
        )
        target = instance.allocate_target(torch.device("cuda"))
        hook_calls: list[nn.Parameter] = []

        def hook(param: nn.Parameter) -> None:
            hook_calls.append(param)
            param.data.add_(1)

        instance.register_post_copy_hook("left.weight", hook)
        instance.load_to_target(target, run_post_copy_hooks=True)

        target_param = target.param_targets["left.weight"].param
        assert pinned.copied == 1
        assert hook_calls == [target_param]
        assert module.left.weight is target_param
        assert module.right.weight is target_param
        torch.testing.assert_close(target_param, torch.full((2, 2), 2.0))

    def test_load_to_target_skips_registered_hooks_by_default(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.zeros(2), requires_grad=False)
        pinned = _FakePinnedParam(torch.ones(2))
        store = PinnedModuleStore(
            params={"weight": cast(PinnedParam, pinned)},
            buffers={},
        )
        instance = PinnedModuleInstance(
            module=module,
            params=store.params,
            buffers=store.buffers,
            cpu_params_by_pinned_id={id(pinned): pinned.make_cpu_param()},
        )
        target = instance.allocate_target(torch.device("cuda"))
        hook_calls: list[nn.Parameter] = []

        def hook(param: nn.Parameter) -> None:
            hook_calls.append(param)
            param.data.add_(1)

        instance.register_post_copy_hook("weight", hook)
        instance.load_to_target(target)

        target_param = target.param_targets["weight"].param
        assert hook_calls == []
        assert module.weight is target_param
        torch.testing.assert_close(target_param, torch.ones(2))

    def test_load_to_target_preserves_trainable_param_wrapper(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.zeros(2), requires_grad=True)
        original = module.weight
        pinned = _FakePinnedParam(torch.ones(2), requires_grad=True)
        store = PinnedModuleStore(
            params={"weight": cast(PinnedParam, pinned)},
            buffers={},
        )
        instance = PinnedModuleInstance(
            module=module,
            params=store.params,
            buffers=store.buffers,
            cpu_params_by_pinned_id={id(pinned): pinned.make_cpu_param()},
        )
        target = instance.allocate_target(torch.device("cuda"))

        instance.load_to_target(target)

        assert module.weight is original
        assert module.weight.data_ptr() == target.param_targets["weight"].param.data_ptr()
        assert pinned.validated == 0

    def test_bind_does_not_revalidate_trainable_param_swap(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.zeros(2), requires_grad=True)
        pinned = _FakePinnedParam(torch.ones(2), requires_grad=True)
        store = PinnedModuleStore(
            params={"weight": cast(PinnedParam, pinned)},
            buffers={},
        )

        store.bind(module)

        assert pinned.validated == 0

    def test_load_to_target_copies_buffers_and_preserves_persistence(self) -> None:
        prototype = nn.Module()
        shared = torch.tensor([1.0, 2.0])
        prototype.register_buffer("running", shared)
        prototype.register_buffer("running_alias", shared)
        store = PinnedModuleStore.from_module(prototype)
        module = nn.Module()
        module.register_buffer("running", torch.zeros(2), persistent=False)
        module.register_buffer("running_alias", module.running, persistent=False)
        instance = store.bind(module)
        target_tensor = torch.empty_like(store.buffers["running"].tensor)
        buffer_target = PinnedBufferTarget(target_tensor)
        target = PinnedModuleTarget(
            param_targets={},
            buffer_targets={
                "running": buffer_target,
                "running_alias": buffer_target,
            },
        )

        instance.load_to_target(target)

        torch.testing.assert_close(target_tensor, store.buffers["running"].tensor)
        assert module.running is target_tensor
        assert module.running_alias is target_tensor
        assert "running" in module._non_persistent_buffers_set
        assert "running_alias" in module._non_persistent_buffers_set

    def test_load_to_target_rejects_unknown_param_targets_before_copying(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.zeros(2), requires_grad=False)
        original = module.weight
        pinned = _FakePinnedParam(torch.ones(2))
        store = PinnedModuleStore(
            params={"weight": cast(PinnedParam, pinned)},
            buffers={},
        )
        instance = PinnedModuleInstance(
            module=module,
            params=store.params,
            buffers=store.buffers,
            cpu_params_by_pinned_id={id(pinned): pinned.make_cpu_param()},
        )
        target = PinnedModuleTarget(
            param_targets={
                "extra": PinnedParamTarget(
                    object(),
                    nn.Parameter(torch.empty(2)),
                ),
            },
            buffer_targets={},
        )

        with pytest.raises(ValueError, match="entries outside the store.*'extra'"):
            instance.load_to_target(target)

        assert pinned.copied == 0
        assert module.weight is original

    def test_load_to_target_rejects_unknown_buffer_targets_before_copying(self) -> None:
        module = nn.Module()
        module.register_buffer("running", torch.zeros(2))
        original = module.running
        pinned = PinnedBuffer.clone(torch.ones(2))
        store = PinnedModuleStore(
            params={},
            buffers={"running": pinned},
        )
        instance = PinnedModuleInstance(
            module=module,
            params=store.params,
            buffers=store.buffers,
            cpu_params_by_pinned_id={},
        )
        target = PinnedModuleTarget(
            param_targets={},
            buffer_targets={
                "running": PinnedBufferTarget(torch.empty_like(pinned.tensor)),
                "extra": PinnedBufferTarget(torch.empty(2)),
            },
        )

        with pytest.raises(ValueError, match="entries outside the store.*'extra'"):
            instance.load_to_target(target)

        assert module.running is original
        assert "extra" not in module._buffers

    def test_copy_trainables_from_target_copies_once_for_aliases(self) -> None:
        pinned = _FakePinnedParam(torch.ones(2), requires_grad=True)
        store = PinnedModuleStore(
            params={
                "left.weight": cast(PinnedParam, pinned),
                "right.weight": cast(PinnedParam, pinned),
            },
            buffers={},
        )
        instance = PinnedModuleInstance(
            module=nn.Module(),
            params=store.params,
            buffers=store.buffers,
            cpu_params_by_pinned_id={id(pinned): pinned.make_cpu_param()},
        )
        target = instance.allocate_target(torch.device("cuda"))

        instance.copy_trainables_from_target(target, non_blocking=True)

        assert pinned.copied_back == 1
        assert pinned.copy_to_cpu_non_blocking == [True]

    def test_copy_trainables_from_target_skips_frozen_params(self) -> None:
        frozen = _FakePinnedParam(torch.ones(2), requires_grad=False)
        trainable = _FakePinnedParam(torch.ones(2), requires_grad=True)
        store = PinnedModuleStore(
            params={
                "frozen": cast(PinnedParam, frozen),
                "trainable": cast(PinnedParam, trainable),
            },
            buffers={},
        )
        instance = PinnedModuleInstance(
            module=nn.Module(),
            params=store.params,
            buffers=store.buffers,
            cpu_params_by_pinned_id={
                id(frozen): frozen.make_cpu_param(),
                id(trainable): trainable.make_cpu_param(),
            },
        )
        target = instance.allocate_target(torch.device("cuda"))

        instance.copy_trainables_from_target(target)

        assert frozen.copied_back == 0
        assert trainable.copied_back == 1

    def test_copy_trainables_from_target_accepts_trainable_only_target(self) -> None:
        frozen = _FakePinnedParam(torch.ones(2), requires_grad=False)
        trainable = _FakePinnedParam(torch.ones(2), requires_grad=True)
        store = PinnedModuleStore(
            params={
                "frozen": cast(PinnedParam, frozen),
                "trainable": cast(PinnedParam, trainable),
            },
            buffers={},
        )
        instance = PinnedModuleInstance(
            module=nn.Module(),
            params=store.params,
            buffers=store.buffers,
            cpu_params_by_pinned_id={
                id(frozen): frozen.make_cpu_param(),
                id(trainable): trainable.make_cpu_param(),
            },
        )
        target = instance.allocate_target(
            torch.device("cuda"),
            param_names={"trainable"},
            buffer_names=(),
        )

        instance.copy_trainables_from_target(target)

        assert frozen.copied_back == 0
        assert trainable.copied_back == 1

    def test_copy_trainables_from_target_validates_before_copying(self) -> None:
        pinned = _FakePinnedParam(torch.ones(2), requires_grad=True)
        store = PinnedModuleStore(
            params={"weight": cast(PinnedParam, pinned)},
            buffers={},
        )
        instance = PinnedModuleInstance(
            module=nn.Module(),
            params=store.params,
            buffers=store.buffers,
            cpu_params_by_pinned_id={id(pinned): pinned.make_cpu_param()},
        )

        with pytest.raises(ValueError, match="param target names mismatch"):
            instance.copy_trainables_from_target(
                PinnedModuleTarget(param_targets={}, buffer_targets={}),
            )

        assert pinned.copied_back == 0

    def test_load_to_target_loads_only_selected_entries(self) -> None:
        module = nn.Module()
        module.frozen = nn.Parameter(torch.zeros(2), requires_grad=False)
        module.trainable = nn.Parameter(torch.zeros(2), requires_grad=True)
        original_frozen = module.frozen
        original_trainable = module.trainable
        frozen = _FakePinnedParam(torch.ones(2), requires_grad=False)
        trainable = _FakePinnedParam(torch.full((2,), 2.0), requires_grad=True)
        store = PinnedModuleStore(
            params={
                "frozen": cast(PinnedParam, frozen),
                "trainable": cast(PinnedParam, trainable),
            },
            buffers={},
        )
        instance = PinnedModuleInstance(
            module=module,
            params=store.params,
            buffers=store.buffers,
            cpu_params_by_pinned_id={
                id(frozen): frozen.make_cpu_param(),
                id(trainable): trainable.make_cpu_param(),
            },
        )

        target = instance.allocate_target(
            torch.device("cuda"),
            param_names=store.trainable_param_names,
            buffer_names=(),
        )
        instance.load_to_target(target, non_blocking=True)

        assert set(target.param_targets) == {"trainable"}
        assert target.buffer_targets == {}
        assert frozen.allocated == 0
        assert trainable.allocated == 1
        assert frozen.copied == 0
        assert trainable.copied == 1
        assert frozen.validated == 0
        assert trainable.validated == 0
        assert module.frozen is original_frozen
        assert module.trainable is original_trainable
        assert module.trainable.data_ptr() == (
            target.param_targets["trainable"].param.data_ptr()
        )

    def test_restore_pinned_restores_partially_loaded_trainables(self) -> None:
        module = nn.Module()
        module.frozen = nn.Parameter(torch.zeros(2), requires_grad=False)
        module.trainable = nn.Parameter(torch.zeros(2), requires_grad=True)
        frozen = _FakePinnedParam(torch.ones(2), requires_grad=False)
        trainable = _FakePinnedParam(torch.full((2,), 2.0), requires_grad=True)
        store = PinnedModuleStore(
            params={
                "frozen": cast(PinnedParam, frozen),
                "trainable": cast(PinnedParam, trainable),
            },
            buffers={},
        )
        instance = PinnedModuleInstance(
            module=module,
            params=store.params,
            buffers=store.buffers,
            cpu_params_by_pinned_id={
                id(frozen): frozen.make_cpu_param(),
                id(trainable): trainable.make_cpu_param(),
            },
        )
        instance.restore_pinned()
        pinned_frozen = module.frozen
        pinned_trainable = module.trainable
        pinned_trainable_data_ptr = module.trainable.data_ptr()
        target = instance.allocate_target(
            torch.device("cuda"),
            param_names=store.trainable_param_names,
            buffer_names=(),
        )
        instance.load_to_target(target)

        instance.restore_pinned()

        assert module.frozen is pinned_frozen
        assert module.trainable is pinned_trainable
        assert target.param_targets["trainable"].param.data_ptr() != (
            pinned_trainable_data_ptr
        )
        assert module.trainable.data_ptr() == pinned_trainable_data_ptr

    def test_binds_same_store_to_multiple_modules(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        prototype.register_buffer("running", torch.randn(2))
        store = PinnedModuleStore.from_module(prototype)

        first = store.bind(prototype)

        second_module = nn.Module()
        second_module.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        second_module.register_buffer("running", torch.randn(2))
        second = store.bind(second_module)

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

    def test_does_not_store_parent_leaf_state(self) -> None:
        module = nn.Module()
        module.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = PinnedModuleStore.from_module(module)

        instance = store.bind(module)

        assert not hasattr(instance, "parent")
        assert not hasattr(instance, "leaf")
        assert not hasattr(instance, "store")

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
        instance = store.bind(target)

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

        instance = store.bind(target)
        cpu_param = instance.cpu_params_by_pinned_id[id(store.params["weight"])]

        assert target.weight is target_param
        assert target.weight.data_ptr() == cpu_param.data_ptr()

    def test_preserves_target_buffer_persistence(self) -> None:
        prototype = nn.Module()
        prototype.register_buffer("running", torch.randn(2), persistent=True)
        store = PinnedModuleStore.from_module(prototype)

        target = nn.Module()
        target.register_buffer("running", torch.randn(2), persistent=False)

        store.bind(target)

        assert target.running is store.buffers["running"].tensor
        assert "running" in target._non_persistent_buffers_set

    def test_non_contiguous_buffer_can_bind_after_store_restore(self) -> None:
        module = nn.Module()
        source = torch.randn(2, 3).t()
        module.register_buffer("table", source)

        store = PinnedModuleStore.from_module(module)
        instance = store.bind(module)

        assert not source.is_contiguous()
        assert store.buffers["table"].tensor.is_contiguous()
        assert instance.module.table is store.buffers["table"].tensor

    def test_rejects_missing_param_name(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = PinnedModuleStore.from_module(prototype)

        with pytest.raises(ValueError, match="missing pinned names.*weight"):
            store.bind(nn.Module())

    def test_rejects_missing_buffer_name(self) -> None:
        prototype = nn.Module()
        prototype.register_buffer("running", torch.randn(2))
        store = PinnedModuleStore.from_module(prototype)

        with pytest.raises(ValueError, match="missing pinned names.*running"):
            store.bind(nn.Module())

    def test_rejects_requires_grad_mismatch(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = PinnedModuleStore.from_module(prototype)

        target = nn.Module()
        target.weight = nn.Parameter(torch.randn(2, 2), requires_grad=True)

        with pytest.raises(ValueError, match="requires_grad mismatch"):
            store.bind(target)

    def test_rejects_param_layout_mismatch(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = PinnedModuleStore.from_module(prototype)

        target = nn.Module()
        target.weight = nn.Parameter(torch.randn(3, 2), requires_grad=False)

        with pytest.raises(ValueError, match="Param 'weight' layout mismatch"):
            store.bind(target)

    def test_rejects_buffer_layout_mismatch(self) -> None:
        prototype = nn.Module()
        prototype.register_buffer("running", torch.randn(2))
        store = PinnedModuleStore.from_module(prototype)

        target = nn.Module()
        target.register_buffer("running", torch.randn(3))

        with pytest.raises(ValueError, match="Buffer 'running' layout mismatch"):
            store.bind(target)

    def test_bind_allows_param_sharing_mismatch(self) -> None:
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

        store.bind(target)

        assert target.left.weight is target.right.weight

    def test_bind_allows_untracked_param_sharing(self) -> None:
        prototype = nn.Module()
        prototype.weight = nn.Parameter(torch.randn(2, 2), requires_grad=False)
        store = PinnedModuleStore.from_module(prototype)

        target = nn.Module()
        shared = torch.randn(2, 2)
        target.weight = nn.Parameter(shared, requires_grad=False)
        target.extra = nn.Parameter(shared, requires_grad=False)
        extra = target.extra

        store.bind(target)

        assert target.extra is extra
        assert target.weight is not target.extra

    def test_bind_allows_buffer_sharing_mismatch(self) -> None:
        prototype = nn.Module()
        shared = torch.randn(2)
        prototype.register_buffer("running", shared)
        prototype.register_buffer("running_alias", shared)
        store = PinnedModuleStore.from_module(prototype)

        target = nn.Module()
        target.register_buffer("running", torch.randn(2))
        target.register_buffer("running_alias", torch.randn(2))

        store.bind(target)

        assert target.running is target.running_alias
