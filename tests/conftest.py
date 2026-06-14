"""Shared test configuration.

torch-offload is a GPU offloading library — pinned host memory and device
transfers are CUDA features — so most of the suite genuinely needs a GPU.
The CPU-runnable subset (registry/dispatch, specs, caching, dequant math,
layout signatures) is what a CPU-only gate (and CI runner) can cover.

To make ``pytest`` green on a CPU-only machine without hand-marking every
GPU test, a test that reaches a CUDA op when no GPU is present raises
``RuntimeError("... CUDA ...")``; treat that as "needs a GPU" and report it
as skipped rather than failed. On a GPU box nothing is intercepted and the
full suite runs. Tests that intend to assert a CUDA error catch it
themselves, so they are unaffected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from collections.abc import Generator


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cuda: test requires a CUDA GPU")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, None, None]:
    outcome = yield
    if torch.cuda.is_available() or outcome.excinfo is None:
        return
    exc = outcome.excinfo[1]
    if isinstance(exc, RuntimeError) and "CUDA" in str(exc):
        outcome.force_exception(
            pytest.skip.Exception(f"needs a CUDA GPU: {exc}", _use_item_location=True)
        )
