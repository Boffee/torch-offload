"""Internal optional-import module for ``bitsandbytes`` support (4-bit + 8-bit).

Single source of truth for everything this repo depends on from
bitsandbytes — both the 4-bit ``Params4bit`` path (NF4/FP4) and the 8-bit
LLM.int8 ``Int8Params`` path. The two share one optional import and the
:data:`BNB_AVAILABLE` flag; the 8-bit helpers live at the bottom of the
module.

4-bit path (NF4 and FP4, with or without double-quant):

- The ``from bitsandbytes.nn import Params4bit`` optional import and the
  :data:`BNB_AVAILABLE` flag.
- :data:`LAYOUT_ATTRS` / :data:`QUANT_STATE_ATTRS` — the attributes this
  repo reads on a ``Params4bit`` and its ``QuantState``.
- Reconstruction via ``Params4bit.from_prequantized`` and quant-state
  (de)serialization via ``QuantState.as_dict(packed=True)`` — the same
  primitives bitsandbytes itself uses to save/load prequantized weights,
  so movement stays codebook- and nesting-agnostic.
- :func:`dequantize_params_4bit` / :func:`requantize_params_4bit` — the
  pieces used by :class:`~torch_offload.bnb4bit_adapter.Bnb4bitAdapter`
  to expose a dequantize/requantize adapter capability (LoRA merge).

Both pin/move/wrap and dequantize/requantize support consume from here
through :mod:`bnb4bit_adapter`, so the layout assumption only has to be
updated once when bitsandbytes refactors.

Pinned to bitsandbytes' internal layout (validated against 0.49.x). Not
part of the public API.
"""

from __future__ import annotations

from typing import Any

import torch

LAYOUT_ATTRS = ("quant_state", "blocksize", "quant_type")
"""Attributes this repo reads from a ``Params4bit`` tensor."""

QUANT_STATE_ATTRS = ("absmax", "code", "blocksize", "quant_type", "dtype", "shape")
"""Attributes this repo reads from a ``Params4bit.quant_state`` (QuantState).

If bitsandbytes refactors and any of these vanishes, callers that go
through :func:`validate_layout` get a framed :class:`RuntimeError` naming
the missing attribute(s); callers that don't would otherwise fail with a
generic ``AttributeError`` deeper in the access path.
"""


try:
    from bitsandbytes import functional as bnb_functional
    from bitsandbytes.nn import Int8Params, Params4bit

    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    Params4bit: Any = None
    Int8Params: Any = None
    bnb_functional: Any = None


def is_params_4bit(t: object) -> bool:
    """Return whether ``t`` is a bitsandbytes ``Params4bit`` (NF4/FP4)."""
    return BNB_AVAILABLE and isinstance(t, Params4bit)


def require_params_4bit(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated ``Params4bit`` tensor, or raise."""
    if not is_params_4bit(t):
        raise TypeError(
            f"expected bitsandbytes Params4bit, got {type(t).__name__}"
        )
    validate_layout(t)
    return t


def validate_layout(t: torch.Tensor) -> None:
    """Raise if ``t`` is not a fully-quantized ``Params4bit`` exposing
    :data:`LAYOUT_ATTRS` / :data:`QUANT_STATE_ATTRS`.

    The adapter (:meth:`Bnb4bitAdapter.matches`) calls this so a layout
    drift in bitsandbytes — or an un-quantized placeholder that cannot be
    offloaded — is reported uniformly rather than as a generic
    ``AttributeError`` partway through dispatch. Cheap enough to run on
    every dispatch (a handful of ``hasattr`` calls), no caching.
    """
    missing = [a for a in LAYOUT_ATTRS if not hasattr(t, a)]
    quant_state = getattr(t, "quant_state", None)
    if quant_state is None:
        missing.append("quant_state (is None — tensor is not quantized yet)")
    else:
        missing += [
            f"quant_state.{a}"
            for a in QUANT_STATE_ATTRS
            if not hasattr(quant_state, a)
        ]
    if not missing:
        return
    raise RuntimeError(
        f"Params4bit is missing expected attributes {missing!r}; this repo "
        f"is pinned to a layout that exposes {LAYOUT_ATTRS} and "
        f"quant_state.{QUANT_STATE_ATTRS}. bitsandbytes likely refactored "
        "the 4-bit wrapper — upgrade torch-offload to match, or the weight "
        "was never quantized (move the model to CUDA before offloading)."
    )


def quant_stats(t: torch.Tensor) -> dict[str, torch.Tensor]:
    """Packed quant-state as an all-tensor dict.

    Keys are exactly what ``Params4bit.from_prequantized`` consumes:
    ``absmax``, ``quant_map`` (the NF4/FP4 codebook), the nested
    ``nested_absmax`` / ``nested_quant_map`` when double-quant is on, and
    one packed metadata blob keyed ``quant_state.bitsandbytes__{nf4,fp4}``
    holding the non-tensor scalars (blocksize, shape, dtype, offset).
    bitsandbytes returns the blob already on host; the absmax/codebook
    tensors follow the weight's device.
    """
    return require_params_4bit(t).quant_state.as_dict(packed=True)


def metadata_blob_key(stats: dict[str, torch.Tensor]) -> str:
    """Return the single ``quant_state.bitsandbytes__*`` key in ``stats``.

    This is the packed scalar-metadata blob (not a compute buffer); the
    adapter keeps it host-resident and never DMAs it.
    """
    keys = [k for k in stats if "quant_state" in k]
    if len(keys) != 1:
        raise RuntimeError(
            f"expected exactly one packed quant_state blob in {list(stats)!r}; "
            f"found {keys!r}. bitsandbytes likely changed as_dict(packed=True)."
        )
    return keys[0]


def build_params_4bit(
    data: torch.Tensor,
    stats: dict[str, torch.Tensor],
    *,
    device: torch.device,
    requires_grad: bool = False,
) -> Any:  # noqa: ANN401
    """Reconstruct a ``Params4bit`` from raw packed bytes + a packed stats dict.

    Used for both the host (deactivated) and GPU (active) wrappers. When
    ``data`` and the compute tensors in ``stats`` already live on
    ``device``, bitsandbytes wraps them in place (``data.to(device)`` and
    ``absmax.to(device)`` are no-ops that return the same storage), so the
    reconstructed wrapper aliases the caller's pre-allocated buffers and a
    later in-place DMA into them is visible through the wrapper.

    ``from_prequantized`` mutates the stats dict (``pop`` + ``update`` while
    unpacking the metadata blob), so a shallow copy is passed.
    """
    if not BNB_AVAILABLE:
        raise RuntimeError("bitsandbytes is required to build a Params4bit")
    return Params4bit.from_prequantized(
        data=data,
        quantized_stats=dict(stats),
        requires_grad=requires_grad,
        device=device,
    )


def dequantize_params_4bit(t: torch.Tensor) -> torch.Tensor:
    """Return the dense logical value of a ``Params4bit`` as fp32.

    Dequantizes to the logical ``quant_state.shape`` — not the packed
    ``(numel/2, 1)`` storage shape that ``Params4bit.dequantize()`` would
    surface — so the result can be re-quantized symmetrically.
    """
    qt = require_params_4bit(t)
    quant_state = qt.quant_state
    dense = bnb_functional.dequantize_4bit(qt.data, quant_state)
    return dense.reshape(quant_state.shape).to(torch.float32)


def requantize_params_4bit(t: torch.Tensor, *, like: torch.Tensor) -> Any:  # noqa: ANN401
    """Encode dense ``t`` in the same 4-bit layout as ``like``.

    Recomputes block scales from ``t``'s values (4-bit absmax is
    data-dependent, so the scales cannot be reused the way an int8 affine
    scale could), preserving ``like``'s quant type, blocksize, and
    double-quant setting.
    """
    ref = require_params_4bit(like)
    quant_state = ref.quant_state
    if tuple(t.shape) != tuple(quant_state.shape):
        raise ValueError(
            f"Cannot requantize tensor with shape {tuple(t.shape)} like "
            f"Params4bit with logical shape {tuple(quant_state.shape)}."
        )
    qdata, new_quant_state = bnb_functional.quantize_4bit(
        t.reshape(quant_state.shape).to(quant_state.dtype),
        blocksize=quant_state.blocksize,
        compress_statistics=quant_state.nested,
        quant_type=quant_state.quant_type,
    )
    return build_params_4bit(
        qdata,
        new_quant_state.as_dict(packed=True),
        device=qdata.device,
    )


# ---------------------------------------------------------------------------
# bitsandbytes 8-bit (Int8Params / LLM.int8) support
# ---------------------------------------------------------------------------

INT8_LAYOUT_ATTRS = ("CB", "SCB")
"""Attributes this repo reads from a quantized ``Int8Params``.

``CB`` is the row-major int8 weight (and ``weight.data``); ``SCB`` is the
per-output-row float32 scale. Both live on the parameter only *before* the
owning ``Linear8bitLt`` runs its first forward — after that bitsandbytes
migrates them onto the module's ``MatmulLtState`` and nulls them here. The
adapter therefore pins a not-yet-forwarded weight; a post-forward (or
unquantized) weight is reported by :func:`validate_int8_layout`.
"""


def is_int8_params(t: object) -> bool:
    """Return whether ``t`` is a bitsandbytes ``Int8Params`` (LLM.int8)."""
    return BNB_AVAILABLE and isinstance(t, Int8Params)


def require_int8_params(t: torch.Tensor) -> Any:  # noqa: ANN401
    """Return ``t`` as a validated ``Int8Params`` tensor, or raise."""
    if not is_int8_params(t):
        raise TypeError(
            f"expected bitsandbytes Int8Params, got {type(t).__name__}"
        )
    validate_int8_layout(t)
    return t


def validate_int8_layout(t: torch.Tensor) -> None:
    """Raise if ``t`` is not a quantized, not-yet-forwarded ``Int8Params``.

    Unlike 4-bit, the int8 quant state (``CB`` + ``SCB``) lives on the
    parameter only until the owning ``Linear8bitLt`` runs its first
    forward, which migrates it onto the module (``MatmulLtState``) and nulls
    ``CB``/``SCB`` here. A tensor-scoped adapter cannot reach the module, so
    it can only pin a pre-forward weight — this validates that and otherwise
    raises a framed error rather than silently capturing ``None``.
    """
    missing = [
        attr
        for attr in INT8_LAYOUT_ATTRS
        if getattr(t, attr, None) is None
    ]
    if not missing:
        return
    raise RuntimeError(
        f"Int8Params is missing {missing!r}; this repo reads {INT8_LAYOUT_ATTRS}. "
        "Either bitsandbytes refactored the int8 wrapper, or — most likely — "
        "the owning Linear8bitLt already ran a forward, which migrates CB/SCB "
        "onto the module and nulls them on the weight. Pin int8 weights before "
        "the first forward (e.g. build the offload store at load time)."
    )


def build_int8_params(
    cb: torch.Tensor,
    scb: torch.Tensor,
    *,
    requires_grad: bool = False,
) -> Any:  # noqa: ANN401
    """Reconstruct an ``Int8Params`` from raw int8 ``CB`` + float32 ``SCB``.

    Passes ``CB``/``SCB`` explicitly so bitsandbytes wraps them in place
    without re-quantizing — the wrapper aliases the caller's buffers (and
    ``weight.data is CB``), so a later in-place DMA into them is visible
    through the wrapper. ``has_fp16_weights=False`` keeps the weight
    statically int8; ``requires_grad`` must stay ``False`` (assigning int8
    data to a grad-requiring Parameter raises).
    """
    if not BNB_AVAILABLE:
        raise RuntimeError("bitsandbytes is required to build an Int8Params")
    return Int8Params(
        cb,
        requires_grad=requires_grad,
        has_fp16_weights=False,
        CB=cb,
        SCB=scb,
    )


def dequantize_int8_params(t: torch.Tensor) -> torch.Tensor:
    """Return the dense logical value of an ``Int8Params`` as fp32.

    Row-wise affine dequant: ``CB * SCB / 127`` per output row.
    """
    qt = require_int8_params(t)
    scale = (qt.SCB.to(torch.float32) / 127.0).view(-1, 1)
    # int8 * fp32 promotes to fp32 directly; no explicit CB widening copy.
    return qt.CB * scale


def requantize_int8_params(t: torch.Tensor, *, like: torch.Tensor) -> Any:  # noqa: ANN401
    """Encode dense ``t`` in the same int8 layout as ``like``.

    Recomputes the per-row scale from ``t`` via ``int8_vectorwise_quant``;
    ``like`` only supplies the expected shape. Quantizes in fp32 rather than
    fp16: a merged value past fp16 max (65504) would otherwise overflow to
    ``inf``, poisoning the whole row's scale to ``inf`` (and the dequantized
    row to ``NaN``); fp32 also avoids dropping mantissa bits during merge.
    """
    ref = require_int8_params(like)
    if tuple(t.shape) != tuple(ref.CB.shape):
        raise ValueError(
            f"Cannot requantize tensor with shape {tuple(t.shape)} like "
            f"Int8Params with shape {tuple(ref.CB.shape)}."
        )
    cb, scb, _outliers = bnb_functional.int8_vectorwise_quant(t.to(torch.float32))
    return build_int8_params(cb, scb)
