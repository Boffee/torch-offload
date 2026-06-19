"""GPU memory management utilities -- model-agnostic, torch-only.

High-level API:

- :class:`ModelCache` with :class:`ModelSpec` caches model stores and
  creates per-use :class:`ModelOffloader` bindings. Use
  :class:`LoRASpec` with ``ModelCache.use(..., loras=[...])`` to cache
  LoRA resources and apply them during model activation.
  :class:`ObjectSpec` caches general Python objects (tokenizers,
  processors, configs) in the same registry; by default they are
  charged zero bytes and live until explicitly evicted.

Lower-level resource bindings:

- :class:`ModelOffloader` -- whole-model pinned-CPU bulk cache when
  created by ``ModelOffloaderStore.from_module(model).bind(model)``, or
  per-block streaming when the store is constructed with ``blocks_attr``
  and ``num_resident_blocks``. Streaming mode supports optional LoRA merge,
  trainable-parameter support, CUDA prefetch on a secondary stream, and
  activation checkpointing through autograd backward. By default,
  trainable params are managed by
  :class:`PinnedComponent` and stay GPU-resident while active; set
  ``stream_trainable_weights=True`` to stream in-block trainable weights
  and materialize them only around ``optimizer.step()``. That step runs on
  the GPU via the ``optimizer_step()`` context. When *every* trainable the
  optimizer touches is streamed, its ``.grad`` lands on CPU on deactivation,
  so calling ``optimizer.step()`` outside ``use()`` instead runs the
  optimizer on CPU over the pinned weights (state stays on host) — keep such
  trainables in fp32. Non-streamed trainables (the default, or params
  outside ``blocks_attr`` like embeddings/heads) keep their grad on GPU, so
  step those inside ``optimizer_step()``.

- :class:`MpsWeights` -- whole-model CPU->MPS materializer. Use for
  frozen models that should become MPS-resident without retaining a
  separate CPU cache. Construction copies one managed tensor at a time
  and immediately replaces its module registry entry to keep peak host memory
  close to one model plus the current tensor.

The CUDA-oriented :class:`ModelOffloader` shares the underlying
per-parameter pinned storage from
:class:`~torch_offload.pinned_param.PinnedParam` (clone + pin
+ optional quanto ``WeightQBytesTensor`` decomposition, bitsandbytes
4-bit ``Params4bit`` (NF4/FP4) and 8-bit ``Int8Params`` (LLM.int8)
decomposition, GGUF packed weights, TorchAO NVFP4 / MX (MXFP8,
MXFP4) / scaled-FP8 / INT8 / INT4 (tile-packed) packed weights, and
tensor-parallel ``DTensor`` weights wrapping any of the above).

:class:`ModelOffloader` and :class:`MpsWeights` implement the
:class:`ModelStrategy` Protocol —
the plug-in contract for per-use bindings that :class:`ModelCache`
activates.

Package stores make ``cache_bytes`` final before use: for
:class:`ModelOffloader`, store construction pins reusable backing storage
and ``bind(model)`` creates the bound model binding; for self-binding stores
like :class:`MpsWeights`, construction owns that work. ``activate(device)``
then makes the binding usable on the requested device. For
:class:`ModelOffloader`, ``deactivate()`` returns managed tensors to
pinned CPU. For :class:`MpsWeights`,
construction has already materialized the model on MPS, so
``activate('mps')`` and ``deactivate()`` are lifecycle-only.

Pinned construction intentionally optimizes peak host memory. For plain
``torch.Tensor`` parameters, pinning clones into pinned CPU storage and
may immediately repoint the source ``Parameter.data`` at that pinned
clone so the original source storage can be freed before all buffers
finish pinning. This avoids a temporary 2x host-memory peak for
CPU-origin models and promptly frees GPU storage for CUDA-origin models.
If construction raises after pinning has started,
recovery of the partially constructed resource/model is unsupported;
drop those references and rebuild from a fresh model instance.

:class:`ModelOffloader` composes (in order):
  1. A :class:`PinnedComponent` for every non-streamed parameter and
     buffer, including trainables skipped by block streaming.
  2. One :class:`StreamedComponent` per ``blocks_attr`` path when
     streaming is configured.

Optional LoRA merging is requested via :meth:`ModelOffloader.set_loras`
and resolved on activation by installing post-copy hooks for canonical
managed parameter targets. Unknown targets raise during activation. The
hooks run immediately after the owning component copies a base weight
from pinned CPU storage to GPU, so block-streamed and non-block weights
use the same merge path. Merge eligibility is owned by the selected
tensor adapter: plain dense tensors opt into in-place
``addmm_``; structured quantized wrappers can opt into
dequantize/requantize plus ``copy_into`` merge, otherwise use routed
LoRA when their module exposes a compatible logical Linear weight shape
and compute dtype.

:class:`ModelCache` manages cached backing stores with policy-driven
eviction, creates per-use bindings, and owns transactional admission.
Trainable model specs reuse their primary model across sequential
uses and reject concurrent same-key bindings. Custom
:class:`EvictionPolicy`
implementations can replace the default LRU behavior. See its docstring
for design notes.

Compatibility
-------------
- **``torch.compile`` is not supported** for managed modules.
- **Wrap before DDP/FSDP**, not after.
- **Coarse cache concurrency.** :class:`ModelCache` serializes cache
  metadata/lifecycle operations and releases its lock while caller code
  runs inside a use context. Individual yielded model objects are not made
  safe for concurrent same-key execution by the cache.
"""

from .gguf_adapter import GGUFWeight
from .lora import LoRA, LoRATransform
from .merge import merge_lora
from .model_cache import (
    DuplicateModelKeyError,
    EvictionCandidate,
    EvictionContext,
    EvictionPolicy,
    EvictionPolicyError,
    LoRASpec,
    LRUEvictionPolicy,
    ModelCache,
    ModelCacheError,
    ModelInUseError,
    ModelNotRegisteredError,
    ModelSpec,
    ModelTooLargeError,
    ObjectSpec,
    ResourceSpec,
)
from .model_offloader import ModelOffloader, ModelOffloaderStore
from .mps_weights import MpsWeights
from .pinned_component import PinnedComponent, PinnedComponentStore
from .protocols import (
    ModelStrategy,
    ModelStrategyComponent,
    ResourceBinding,
    ResourceStore,
)
from .streamed_component import StreamedComponent, StreamedComponentStore

__all__ = [
    "DuplicateModelKeyError",
    "EvictionCandidate",
    "EvictionContext",
    "EvictionPolicy",
    "EvictionPolicyError",
    "GGUFWeight",
    "LRUEvictionPolicy",
    "LoRA",
    "LoRASpec",
    "LoRATransform",
    "ModelCache",
    "ModelCacheError",
    "ModelInUseError",
    "ModelNotRegisteredError",
    "ModelOffloader",
    "ModelOffloaderStore",
    "ModelSpec",
    "ModelStrategy",
    "ModelStrategyComponent",
    "ModelTooLargeError",
    "MpsWeights",
    "ObjectSpec",
    "PinnedComponent",
    "PinnedComponentStore",
    "ResourceBinding",
    "ResourceSpec",
    "ResourceStore",
    "StreamedComponent",
    "StreamedComponentStore",
    "merge_lora",
]
