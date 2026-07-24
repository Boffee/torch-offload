"""Opt-in ``torch.compile`` integration for streamed blocks."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from torch_offload import (
    BlockCompileConfig,
    LoRA,
    ModelOffloader,
    ModelSpec,
    StreamConfig,
)
from tests.conftest import activated_model, streamed_components

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _Block(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(x))


class _BlockModel(nn.Module):
    def __init__(
        self,
        *,
        num_blocks: int = 2,
        width: int = 8,
        blocks: list[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if blocks is None:
            blocks = [_Block(width) for _ in range(num_blocks)]
        self.blocks = nn.ModuleList(blocks)
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _TwoGroupModel(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.first_blocks = nn.ModuleList([_Block(width), _Block(width)])
        self.second_blocks = nn.ModuleList([_Block(width), _Block(width)])
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.first_blocks:
            x = block(x)
        for block in self.second_blocks:
            x = block(x)
        return x


class _CompileSpy:
    """A lazy ``torch.compile`` stand-in that records construction/execution."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.calls: list[tuple[Callable[..., object], dict[str, object]]] = []
        self.executions = 0
        self._events = events

    def __call__(
        self,
        fn: Callable[..., object],
        **kwargs: object,
    ) -> Callable[..., object]:
        self.calls.append((fn, kwargs))

        def compiled(*args: object, **call_kwargs: object) -> object:
            self.executions += 1
            if self._events is not None:
                self._events.append("compiled")
            return fn(*args, **call_kwargs)

        return compiled


def _make_offloader(
    model: nn.Module,
    *,
    blocks_attr: list[str] | None = None,
    block_compile: BlockCompileConfig | None = None,
) -> ModelOffloader:
    return ModelOffloader.from_module(
        model,
        blocks_attr=["blocks"] if blocks_attr is None else blocks_attr,
        block_compile=block_compile,
    )


def _stream_config() -> StreamConfig:
    return StreamConfig(num_resident_blocks=1, num_prefetch_blocks=0)


class TestBlockCompileConfig:
    def test_defaults(self) -> None:
        config = BlockCompileConfig()

        assert config.dynamic is True
        assert config.fullgraph is False

    @pytest.mark.parametrize("dynamic", [0, "yes", object()])
    def test_dynamic_must_be_bool_or_none(self, dynamic: object) -> None:
        with pytest.raises(TypeError, match="dynamic must be bool or None"):
            BlockCompileConfig(dynamic=dynamic)  # type: ignore[arg-type]

    @pytest.mark.parametrize("fullgraph", [0, "yes", object()])
    def test_fullgraph_must_be_bool(self, fullgraph: object) -> None:
        with pytest.raises(TypeError, match="fullgraph must be bool"):
            BlockCompileConfig(fullgraph=fullgraph)  # type: ignore[arg-type]

    def test_compile_requires_blocks_attr(self) -> None:
        model = nn.Linear(4, 4, bias=False)

        with pytest.raises(
            ValueError,
            match="block_compile requires at least one blocks_attr",
        ):
            ModelOffloader.from_module(
                model,
                block_compile=BlockCompileConfig(),
            )

    def test_model_spec_passes_config_to_bound_streamer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        config = BlockCompileConfig(dynamic=None, fullgraph=True)
        spec = ModelSpec(
            key="compiled",
            estimated_cache_bytes=1024,
            factory=_BlockModel,
            blocks_attr=("blocks",),
            block_compile=config,
        )

        offloader = spec.build_store()
        try:
            streamer = streamed_components(offloader)[0]
            assert streamer.block_compile is config
            assert len(spy.calls) == 2
            assert all(
                kwargs
                == {
                    "backend": "inductor",
                    "mode": "default",
                    "dynamic": None,
                    "fullgraph": True,
                }
                for _fn, kwargs in spy.calls
            )
        finally:
            offloader.deactivate()

    def test_compile_policy_does_not_change_cache_bytes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(torch, "compile", _CompileSpy())
        eager = _make_offloader(_BlockModel())
        compiled = _make_offloader(
            _BlockModel(),
            block_compile=BlockCompileConfig(),
        )
        try:
            assert compiled.cache_bytes == eager.cache_bytes
        finally:
            compiled.deactivate()
            eager.deactivate()


class TestCompiledForwardConstruction:
    def test_aliased_block_module_is_compiled_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        shared = _Block()
        model = _BlockModel(blocks=[shared, shared])

        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        try:
            assert len(spy.calls) == 1
        finally:
            offloader.deactivate()

    def test_one_config_applies_to_multiple_block_groups(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        model = _TwoGroupModel()
        config = BlockCompileConfig()

        offloader = _make_offloader(
            model,
            blocks_attr=["first_blocks", "second_blocks"],
            block_compile=config,
        )
        try:
            streamers = streamed_components(offloader)
            assert len(streamers) == 2
            assert all(streamer.block_compile is config for streamer in streamers)
            assert len(spy.calls) == 4
        finally:
            offloader.deactivate()


class TestCompiledForwardLifecycle:
    def test_cpu_activation_remains_eager(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        model = _BlockModel()
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        try:
            with activated_model(offloader, "cpu"):
                with torch.inference_mode():
                    model(torch.randn(2, 8))
                assert all("forward" not in block.__dict__ for block in model.blocks)
            assert spy.executions == 0
        finally:
            offloader.deactivate()

    @CUDA
    def test_cuda_installs_after_stream_hook_and_restores_descriptor_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        events: list[str] = []
        spy = _CompileSpy(events)
        monkeypatch.setattr(torch, "compile", spy)
        model = _BlockModel()
        assert all("forward" not in block.__dict__ for block in model.blocks)
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        streamer = streamed_components(offloader)[0]
        original_before = streamer._before_block_forward

        def record_before(*args: object, **kwargs: object) -> None:
            events.append("stream")
            original_before(*args, **kwargs)

        monkeypatch.setattr(streamer, "_before_block_forward", record_before)
        try:
            with activated_model(
                offloader,
                "cuda",
                stream_config=_stream_config(),
            ):
                assert all("forward" in block.__dict__ for block in model.blocks)
                with torch.inference_mode():
                    model(torch.randn(2, 8, device="cuda"))
            assert events == ["stream", "compiled", "stream", "compiled"]
            assert all("forward" not in block.__dict__ for block in model.blocks)
        finally:
            offloader.deactivate()

    @CUDA
    def test_existing_instance_forward_override_is_restored_verbatim(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        block = _Block()
        class_forward = block.forward

        def override(x: torch.Tensor) -> torch.Tensor:
            return class_forward(x) + 1

        block.forward = override
        original_override = block.__dict__["forward"]
        model = _BlockModel(blocks=[block])
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        try:
            with activated_model(
                offloader,
                "cuda",
                stream_config=_stream_config(),
            ):
                assert block.__dict__["forward"] is not original_override
            assert block.__dict__["forward"] is original_override
        finally:
            offloader.deactivate()

    @CUDA
    def test_activation_failure_restores_original_forwards(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(torch, "compile", _CompileSpy())
        model = _BlockModel()
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        streamer = streamed_components(offloader)[0]
        compile_state = streamer._block_compile
        state_type = type(compile_state)
        original_install = state_type.install

        def broken_install(
            state: object,
            active_config: BlockCompileConfig | None,
        ) -> None:
            original_install(state, active_config)
            raise RuntimeError("simulated compiled-forward install failure")

        with monkeypatch.context() as install_patch:
            install_patch.setattr(state_type, "install", broken_install)
            with pytest.raises(RuntimeError, match="simulated compiled-forward"):
                offloader.activate(
                    "cuda",
                    stream_config=_stream_config(),
                )

        assert offloader.active_device is None
        assert not compile_state.installed
        assert all("forward" not in block.__dict__ for block in model.blocks)

        with activated_model(
            offloader,
            "cuda",
            stream_config=_stream_config(),
        ):
            pass

    @CUDA
    def test_real_inductor_supports_dynamic_shapes_and_reactivation(self) -> None:
        torch.manual_seed(0)
        model = _BlockModel()
        inputs = [
            torch.randn(2, 8),
            torch.randn(3, 8),
            torch.randn(4, 8),
        ]
        with torch.inference_mode():
            expected = [model(x) for x in inputs]

        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        try:
            with activated_model(
                offloader,
                "cuda",
                stream_config=_stream_config(),
            ):
                with torch.inference_mode():
                    actual = [model(x.cuda()).cpu() for x in inputs[:2]]
            with activated_model(
                offloader,
                "cuda",
                stream_config=_stream_config(),
            ):
                with torch.inference_mode():
                    actual.append(model(inputs[2].cuda()).cpu())

            for actual_value, expected_value in zip(
                actual,
                expected,
                strict=True,
            ):
                torch.testing.assert_close(
                    actual_value,
                    expected_value,
                    rtol=1e-4,
                    atol=1e-5,
                )
        finally:
            offloader.deactivate()


class TestCompileFailureSemantics:
    @CUDA
    def test_compiler_failure_propagates_without_eager_retry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eager_calls = 0

        class CountingBlock(_Block):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                nonlocal eager_calls
                eager_calls += 1
                return super().forward(x)

        def failing_compile(
            _fn: Callable[..., object],
            **_kwargs: object,
        ) -> Callable[..., object]:
            def fail(*_args: object, **_call_kwargs: object) -> object:
                raise RuntimeError("simulated compiler failure")

            return fail

        monkeypatch.setattr(torch, "compile", failing_compile)
        model = _BlockModel(blocks=[CountingBlock()])
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        try:
            with activated_model(
                offloader,
                "cuda",
                stream_config=_stream_config(),
            ):
                with pytest.raises(RuntimeError, match="simulated compiler"):
                    model(torch.randn(2, 8, device="cuda"))
            assert eager_calls == 0
        finally:
            offloader.deactivate()

    @CUDA
    def test_model_exception_propagates_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eager_calls = 0

        class FailingBlock(_Block):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                nonlocal eager_calls
                eager_calls += 1
                raise ValueError("model forward failed")

        monkeypatch.setattr(torch, "compile", _CompileSpy())
        model = _BlockModel(blocks=[FailingBlock()])
        offloader = _make_offloader(
            model,
            block_compile=BlockCompileConfig(),
        )
        try:
            with activated_model(
                offloader,
                "cuda",
                stream_config=_stream_config(),
            ):
                with pytest.raises(ValueError, match="model forward failed"):
                    model(torch.randn(2, 8, device="cuda"))
            assert eager_calls == 1
        finally:
            offloader.deactivate()


class TestCompiledLoRA:
    @CUDA
    def test_real_inductor_matches_eager_merge_mode(self) -> None:
        torch.manual_seed(0)
        eager_model = _BlockModel()
        compiled_model = _BlockModel()
        compiled_model.load_state_dict(eager_model.state_dict())
        lora = LoRA.from_state_dict(
            {
                "blocks.0.proj.lora_A.weight": torch.randn(2, 8),
                "blocks.0.proj.lora_B.weight": torch.randn(8, 2),
            }
        )
        x = torch.randn(2, 8, device="cuda")

        eager_offloader = _make_offloader(eager_model)
        try:
            with activated_model(
                eager_offloader,
                "cuda",
                loras=[lora],
                lora_strengths=[0.25],
                lora_mode="merge",
                stream_config=_stream_config(),
            ):
                with torch.inference_mode():
                    expected = eager_model(x).cpu()
        finally:
            eager_offloader.deactivate()

        compiled_offloader = _make_offloader(
            compiled_model,
            block_compile=BlockCompileConfig(),
        )
        try:
            with activated_model(
                compiled_offloader,
                "cuda",
                loras=[lora],
                lora_strengths=[0.25],
                lora_mode="merge",
                stream_config=_stream_config(),
            ):
                with torch.inference_mode():
                    actual = compiled_model(x).cpu()
        finally:
            compiled_offloader.deactivate()

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)

    @CUDA
    def test_routed_bypass_is_model_wide_and_temporary(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        spy = _CompileSpy()
        monkeypatch.setattr(torch, "compile", spy)
        model = _TwoGroupModel()
        lora = LoRA.from_state_dict(
            {
                "first_blocks.0.proj.lora_A.weight": torch.randn(2, 8),
                "first_blocks.0.proj.lora_B.weight": torch.randn(8, 2),
            }
        )
        offloader = _make_offloader(
            model,
            blocks_attr=["first_blocks", "second_blocks"],
            block_compile=BlockCompileConfig(),
        )
        x = torch.randn(2, 8, device="cuda")
        try:
            with activated_model(
                offloader,
                "cuda",
                loras=[lora],
                lora_mode="routed",
                stream_config=_stream_config(),
            ):
                assert all("forward" not in block.__dict__ for block in (*model.first_blocks, *model.second_blocks))
                with torch.inference_mode():
                    model(x)
            assert spy.executions == 0

            with activated_model(
                offloader,
                "cuda",
                lora_mode="routed",
                stream_config=_stream_config(),
            ):
                with torch.inference_mode():
                    model(x)
            assert spy.executions == 4

            with activated_model(
                offloader,
                "cuda",
                loras=[lora],
                lora_mode="merge",
                stream_config=_stream_config(),
            ):
                with torch.inference_mode():
                    model(x)
            assert spy.executions == 8
        finally:
            offloader.deactivate()
