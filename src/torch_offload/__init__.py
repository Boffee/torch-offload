"""GPU memory management utilities — model-agnostic, torch-only.

Two complementary offload strategies:

- :class:`ModelOffloader` — per-block streaming with optional LoRA
  merge and trainable-parameter support. Use for models whose
  individual blocks fit on GPU but the whole model does not.
  Hooks-based, prefetches upcoming blocks on a secondary CUDA stream,
  supports gradient checkpointing through autograd backward. By
  default, trainable params stay GPU-resident while active; set
  ``stream_trainable_weights=True`` to stream in-block trainable
  weights and materialize them only around ``optimizer.step()``. Composes
  :class:`PinnedWeights` + :class:`TrainableWeights` +
  :class:`StreamedWeights` internally.

- :class:`PinnedWeights` — whole-model pinned-CPU bulk cache. Use for
  models that fit on GPU when active but should be evicted between
  calls (e.g., text encoder during diffusion). One CPU->GPU transfer
  per use; on deactivate, parameter slots are repointed at pinned
  CPU storage and the GPU storage is released by refcount.

Both classes share the underlying per-parameter pinned storage from
:class:`~torch_offload.pinned_buffer.PinnedParamBuffer` (clone + pin
+ optional quanto ``WeightQBytesTensor`` decomposition, GGUF packed
weights, and TorchAO NVFP4 packed weights), so quantized models work
with either.

Both :class:`PinnedWeights` and :class:`ModelOffloader` implement the
:class:`ModelStrategy` Protocol — the plug-in contract for
storage/placement strategies that :class:`ModelCache` consumes.

All strategies pin in their constructor, so ``cache_bytes`` is final
immediately and :class:`ModelCache` can admit them without a
factory-side ``prepare()`` dance. ``activate(device)`` then brings
everything to the requested device; ``deactivate()`` returns to pinned
CPU.

Construction intentionally optimizes peak host memory. For plain
``torch.Tensor`` parameters, pinning clones into pinned CPU storage and
may immediately repoint the source ``Parameter.data`` at that pinned
clone so the original pageable storage can be freed before all buffers
finish pinning. If construction raises after pinning has started,
recovery of the partially constructed strategy/model is unsupported;
drop those references and rebuild from a fresh model instance.

:class:`ModelOffloader` composes (in order):
  1. A non-block :class:`PinnedWeights` with a :class:`SlotOwnership`
     skip filter for everything outside the block list.
  2. A :class:`TrainableWeights` for LoRA / adapter weights
     (or only out-of-block trainables when
     ``stream_trainable_weights=True``).
  3. One :class:`StreamedWeights` per ``layers_attr`` path.

Optional LoRA merging is requested via :meth:`ModelOffloader.set_loras`
and resolved on activation by installing post-copy hooks for the matched
targets. The hooks run immediately after the owning component copies a
base weight from pinned CPU storage to GPU, so block-streamed and
non-block weights use the same merge path. Merge eligibility is owned by
the selected tensor adapter: plain dense tensors opt into in-place
``addmm_``; structured quantized wrappers should use routed LoRA only
when their module exposes a compatible logical Linear weight shape and
compute dtype, unless they provide a separate permanent merge path.

Cross-region tied parameters (block <-> non-block, cross-block, or
mixed trainable/frozen across regions) are detected at construction
and raise — slot-local block streaming cannot preserve such ties; use
whole-model :class:`PinnedWeights` instead.

:class:`ModelCache` manages the cached backing storage of multiple
strategies with policy-driven eviction, an active-set with refcounted
leases, and transactional admission. Custom :class:`EvictionPolicy`
and :class:`PlacementPolicy` implementations can replace the default
LRU and one-model-per-CUDA-device policies. See its docstring for
design notes.

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
    GpuDeviceOccupiedError,
    LRUEvictionPolicy,
    ModelCache,
    ModelCacheError,
    ModelInUseError,
    ModelNotRegisteredError,
    ModelSpec,
    ModelTooLargeError,
    OneModelPerCudaDevicePolicy,
    PlacementLease,
    PlacementPolicy,
    ResourceSpec,
)
from .model_offloader import ModelOffloader, detect_streaming_region_ties
from .pinned_weights import PinnedWeights
from .protocols import CachedResource, ModelStrategy, ModelStrategyComponent, SlotOwnership
from .streamed_weights import StreamedWeights
from .trainable_weights import TrainableWeights

__all__ = [
    "ActivationError",
    "CachedResource",
    "DuplicateModelKeyError",
    "EvictionCandidate",
    "EvictionContext",
    "EvictionPolicy",
    "EvictionPolicyError",
    "GGUFWeight",
    "GpuDeviceOccupiedError",
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
    "OneModelPerCudaDevicePolicy",
    "PinnedWeights",
    "PlacementLease",
    "PlacementPolicy",
    "ResourceSpec",
    "SlotOwnership",
    "StreamedWeights",
    "TrainableWeights",
    "detect_streaming_region_ties",
    "merge_lora",
]
