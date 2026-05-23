"""GPU memory management utilities — model-agnostic, torch-only.

Top-level offload strategies:

- :class:`ModelOffloader` — whole-model pinned-CPU bulk cache when
  constructed as ``ModelOffloader(model)``, or per-block streaming when
  constructed with ``layers_attr`` and ``blocks_to_swap``. Streaming mode
  supports optional LoRA merge, trainable-parameter support, CUDA
  prefetch on a secondary stream, and activation checkpointing through
  autograd backward. By default, trainable params are managed by
  :class:`PinnedComponent` and stay GPU-resident while active; set
  ``stream_trainable_weights=True`` to stream in-block trainable weights
  and materialize them only around ``optimizer.step()``.

- :class:`MpsWeights` — whole-model CPU->MPS materializer. Use for
  frozen models that should become MPS-resident without retaining a
  separate CPU cache. Construction copies one managed tensor at a time
  and immediately replaces its module registry entry to keep peak host memory
  close to one model plus the current tensor.

The CUDA-oriented :class:`ModelOffloader` shares the underlying
per-parameter pinned storage from
:class:`~torch_offload.pinned_param.PinnedParam` (clone + pin
+ optional quanto ``WeightQBytesTensor`` decomposition, GGUF packed
weights, and TorchAO NVFP4 packed weights).

:class:`ModelOffloader` and :class:`MpsWeights` implement the
:class:`ModelStrategy` Protocol —
the plug-in contract for storage strategies that :class:`ModelCache`
consumes.

Package strategies make ``cache_bytes`` final in their constructor, so
:class:`ModelCache` can admit them without a factory-side ``prepare()``
dance. ``activate(device)`` then makes the resource usable on the
requested device. For :class:`ModelOffloader`, ``deactivate()`` returns
managed tensors to pinned CPU. For :class:`MpsWeights`,
construction has already materialized the model on MPS, so
``activate('mps')`` and ``deactivate()`` are lifecycle-only.

Pinned construction intentionally optimizes peak host memory. For plain
``torch.Tensor`` parameters, pinning clones into pinned CPU storage and
may immediately repoint the source ``Parameter.data`` at that pinned
clone so the original source storage can be freed before all buffers
finish pinning. This avoids a temporary 2x host-memory peak for
CPU-origin models and promptly frees GPU storage for CUDA-origin models.
If construction raises after pinning has started,
recovery of the partially constructed strategy/model is unsupported;
drop those references and rebuild from a fresh model instance.

:class:`ModelOffloader` composes (in order):
  1. A :class:`PinnedComponent` for every non-streamed parameter and
     buffer, including trainables skipped by block streaming.
  2. One :class:`StreamedComponent` per ``layers_attr`` path when
     streaming is configured.

Optional LoRA merging is requested via :meth:`ModelOffloader.set_loras`
and resolved on activation by installing post-copy hooks for the matched
targets. The hooks run immediately after the owning component copies a
base weight from pinned CPU storage to GPU, so block-streamed and
non-block weights use the same merge path. Merge eligibility is owned by
the selected tensor adapter: plain dense tensors opt into in-place
``addmm_``; structured quantized wrappers can opt into
dequantize/requantize plus ``copy_into`` merge, otherwise use routed
LoRA when their module exposes a compatible logical Linear weight shape
and compute dtype.

:class:`ModelCache` manages the cached backing storage of multiple
strategies with policy-driven eviction, an active-set with refcounted
leases, and transactional admission. Custom :class:`EvictionPolicy`
implementations can replace the default LRU behavior. See its docstring
for design notes.

Compatibility
-------------
- **``torch.compile`` is not supported** for managed modules.
- **Wrap before DDP/FSDP**, not after.
- **Coarse cache concurrency.** :class:`ModelCache` serializes cache
  metadata/lifecycle operations and releases its lock while caller code
  runs inside a lease. Individual yielded model objects are not made
  safe for concurrent same-key execution by the cache.
"""

from .gguf_adapter import GGUFWeight
from .lora import LoRA, LoRATransform
from .merge import merge_lora
from .model_cache import (
    ActivationError,
    DuplicateModelKeyError,
    EvictionCandidate,
    EvictionContext,
    EvictionPolicy,
    EvictionPolicyError,
    LRUEvictionPolicy,
    ModelCache,
    ModelCacheError,
    ModelInUseError,
    ModelNotRegisteredError,
    ModelSpec,
    ModelTooLargeError,
    ResourceSpec,
)
from .model_offloader import ModelOffloader
from .mps_weights import MpsWeights
from .pinned_component import PinnedComponent, PinnedComponentStore
from .protocols import CachedResource, ModelStrategy, ModelStrategyComponent
from .streamed_component import StreamedComponent

__all__ = [
    "ActivationError",
    "CachedResource",
    "DuplicateModelKeyError",
    "EvictionCandidate",
    "EvictionContext",
    "EvictionPolicy",
    "EvictionPolicyError",
    "GGUFWeight",
    "LRUEvictionPolicy",
    "LoRA",
    "LoRATransform",
    "ModelCache",
    "ModelCacheError",
    "ModelInUseError",
    "ModelNotRegisteredError",
    "ModelOffloader",
    "ModelSpec",
    "ModelStrategy",
    "ModelStrategyComponent",
    "ModelTooLargeError",
    "MpsWeights",
    "PinnedComponent",
    "PinnedComponentStore",
    "ResourceSpec",
    "StreamedComponent",
    "merge_lora",
]
