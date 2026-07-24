"""Tests for public external tensor-adapter registration."""

from __future__ import annotations

import subprocess
import sys
from typing import cast

import pytest
import torch

from torch_offload import TensorAdapter, register_adapter
from torch_offload.dtensor_adapter import DTensorAdapter
from torch_offload.tensor_adapter_registry import select_adapter, tensor_id
from torch_offload.tensor_adapters import RegularAdapter


def test_package_import_does_not_require_triton() -> None:
    script = (
        "import builtins\n"
        "real_import = builtins.__import__\n"
        "def import_without_triton("
        "name, globals=None, locals=None, fromlist=(), level=0):\n"
        "    if name == 'triton' or name.startswith('triton.'):\n"
        "        raise ModuleNotFoundError("
        "'No module named triton', name='triton')\n"
        "    return real_import(name, globals, locals, fromlist, level)\n"
        "builtins.__import__ = import_without_triton\n"
        "import torch_offload\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


class _ExternalTensor(torch.Tensor):
    pass


def _external_tensor() -> _ExternalTensor:
    return cast(
        _ExternalTensor,
        torch.Tensor._make_subclass(
            _ExternalTensor,
            torch.arange(4, dtype=torch.float32),
            False,
        ),
    )


class _ExternalAdapter(RegularAdapter):
    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        return isinstance(t, _ExternalTensor)

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple[object, ...]:
        raw = t.as_subclass(torch.Tensor)
        return (
            "external",
            raw.device,
            raw.data_ptr(),
            raw.dtype,
            tuple(raw.shape),
            raw.stride(),
            raw.storage_offset(),
        )


class _PlainOverrideAdapter(RegularAdapter):
    @staticmethod
    def matches(t: torch.Tensor) -> bool:
        return type(t) is torch.Tensor

    @staticmethod
    def tensor_id(t: torch.Tensor) -> tuple[object, ...]:
        return ("external-plain", *RegularAdapter.tensor_id(t)[1:])


class _OlderExternalAdapter(_ExternalAdapter):
    pass


class _NewerExternalAdapter(_ExternalAdapter):
    pass


def test_registers_custom_subclass_for_selection_and_identity() -> None:
    tensor = _external_tensor()
    with pytest.raises(NotImplementedError, match="register_adapter"):
        select_adapter(tensor)

    remove_adapter = register_adapter(_ExternalAdapter)
    try:
        assert isinstance(select_adapter(tensor), _ExternalAdapter)
        assert tensor_id(tensor)[0] == "external"
    finally:
        remove_adapter()

    with pytest.raises(NotImplementedError, match="register_adapter"):
        select_adapter(tensor)


def test_registered_adapter_precedes_builtin_match() -> None:
    tensor = torch.arange(4, dtype=torch.float32)
    assert isinstance(select_adapter(tensor), RegularAdapter)

    remove_adapter = register_adapter(_PlainOverrideAdapter)
    try:
        assert isinstance(select_adapter(tensor), _PlainOverrideAdapter)
        assert tensor_id(tensor)[0] == "external-plain"
    finally:
        remove_adapter()

    assert isinstance(select_adapter(tensor), RegularAdapter)


def test_newest_registration_wins_and_removal_reveals_previous() -> None:
    tensor = _external_tensor()
    remove_older = register_adapter(_OlderExternalAdapter)
    try:
        remove_newer = register_adapter(_NewerExternalAdapter)
        try:
            assert isinstance(select_adapter(tensor), _NewerExternalAdapter)
        finally:
            remove_newer()
        assert isinstance(select_adapter(tensor), _OlderExternalAdapter)
    finally:
        remove_older()


def test_removal_is_idempotent_and_allows_reregistration() -> None:
    remove_adapter = register_adapter(_ExternalAdapter)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_adapter(_ExternalAdapter)
    finally:
        remove_adapter()
    remove_adapter()

    remove_again = register_adapter(_ExternalAdapter)
    remove_again()


def test_rejects_builtin_and_incomplete_adapter_classes() -> None:
    class IncompleteAdapter:
        @staticmethod
        def matches(t: torch.Tensor) -> bool:
            return True

    with pytest.raises(ValueError, match="already built in"):
        register_adapter(RegularAdapter)
    with pytest.raises(TypeError, match="TensorAdapter protocol"):
        register_adapter(IncompleteAdapter)  # type: ignore[arg-type]


def test_dtensor_outer_wrapper_precedes_external_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        DTensorAdapter,
        "matches",
        staticmethod(lambda _tensor: True),
    )
    remove_adapter = register_adapter(_PlainOverrideAdapter)
    try:
        assert isinstance(select_adapter(torch.zeros(1)), DTensorAdapter)
    finally:
        remove_adapter()


def test_tensor_adapter_contract_is_public() -> None:
    assert isinstance(_ExternalAdapter(), TensorAdapter)
