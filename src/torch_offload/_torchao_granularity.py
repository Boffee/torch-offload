"""Internal helper: invert TorchAO's ``get_block_size`` to a granularity.

TorchAO's affine-quantized tensors (``Float8Tensor``, ``Int8Tensor``)
store the per-block partition as a ``block_size`` list rather than the
``Granularity`` object it was derived from, but the re-encoding
constructors (``from_hp``) take a ``Granularity``. Recovering it is the
same arithmetic for every such format — it depends only on ``block_size``
vs ``shape`` and the shared ``PerRow`` / ``PerTensor`` classes, not on any
per-format layout — so it lives here once instead of in each
``_torchao_<format>`` module.
"""

from __future__ import annotations

from typing import Any

try:
    from torchao.quantization.granularity import PerRow, PerTensor

    TORCHAO_GRANULARITY_AVAILABLE = True
except ImportError:
    TORCHAO_GRANULARITY_AVAILABLE = False
    PerRow: Any = None
    PerTensor: Any = None


def granularity_from_block_size(
    block_size: tuple[int, ...], shape: tuple[int, ...], *, label: str,
) -> object:
    """Invert TorchAO's ``get_block_size`` for the affine granularities.

    Supports ``PerTensor`` (block covers the whole tensor) and ``PerRow``
    (block covers one axis fully, 1 elsewhere — the standard
    per-output-channel weight recipe gives ``[1, in]``). Shapes where both
    readings coincide (e.g. a dim of size 1) produce identical block
    partitions either way, so any matching reading is correct. ``label``
    names the wrapper type for the error raised on an unrecognized
    granularity.
    """
    if block_size == shape:
        return PerTensor()
    for dim, size in enumerate(shape):
        per_row = tuple(size if i == dim else 1 for i in range(len(shape)))
        if block_size == per_row:
            return PerRow(dim=dim)
    raise ValueError(
        f"{label} block_size {block_size!r} for shape {shape!r} matches "
        "neither PerTensor nor PerRow granularity; TorchAO likely added a "
        "granularity this adapter does not support yet."
    )
