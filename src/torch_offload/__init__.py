"""GPU memory management utilities -- model-agnostic, torch-only.

High-level API:

- :class:`ResourceCache` accepts structural :class:`ResourceSpec` implementations
  and leases reusable model, LoRA, and object stores under a host-memory
  budget. :class:`ModelCache` specializes it with model-aware use: it composes
  a leased :class:`ModelSpec` with optional :class:`LoRASpec` resources and
  activates the cached :class:`ModelOffloader`.
  :class:`ObjectSpec` caches general Python objects (tokenizers,
  processors, configs) in the same registry; by default they are
  charged zero bytes and live until explicitly evicted.

Lower-level resource bindings:

- :class:`ModelOffloader` -- whole-model pinned-CPU bulk cache when
  created by ``ModelOffloader.from_module(model)``, or per-block streaming
  when it is constructed with ``blocks_attr``
  and activation receives a :class:`StreamConfig`. Streaming mode supports optional LoRA merge,
  trainable-parameter support, CUDA prefetch on a secondary stream, and
  activation checkpointing through autograd backward. By default,
  trainable params are managed by
  :class:`PinnedComponent` and stay GPU-resident while active; set
  ``stream_trainable_weights=True`` to stream in-block trainable weights
  and materialize them only around ``optimizer.step()``. That step runs on
  the GPU via the ``optimizer_step()`` context; calling ``optimizer.step()``
  after deactivation instead runs the optimizer on CPU over the pinned
  weights (state stays on host). On deactivation every managed trainable's
  ``.grad`` follows its ``.data`` to pinned CPU, so the context-free CPU
  step works for both streamed and non-streamed trainables — keep such
  trainables in fp32.

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
MXFP4) / dynamic or calibrated-static scaled-FP8 / INT8 / INT4
(tile-packed) packed weights, and
tensor-parallel ``DTensor`` weights wrapping any of the above).

:class:`ModelOffloader` and :class:`MpsWeights` are cached resources that also
implement the :class:`ResourceBinding` Protocol. Each owns exactly one model
runtime and is reused sequentially.

Package resources make ``cache_bytes`` final during construction.
``activate(device)`` then makes the resource usable on the requested device. For
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

Optional LoRA merging is requested directly on :meth:`ModelOffloader.activate`
and resolved by installing post-copy hooks for managed parameter targets.
Unknown targets raise during activation. The
hooks run immediately after the owning component copies a base weight
from pinned CPU storage to GPU, so block-streamed and non-block weights
use the same merge path. Merge eligibility is owned by the selected
tensor adapter: plain dense tensors opt into in-place
``addmm_``; structured quantized wrappers can opt into
dequantize/requantize plus ``copy_into`` merge, otherwise use routed
LoRA when their module exposes a compatible logical Linear weight shape
and compute dtype.

:class:`LoRA` owns immutable pinned factor storage. Merge and routed consumers
read that backing directly and may overlap; routed hooks stage their own
per-forward device copies.

Downstream tensor subclasses can participate in pinning and movement without
adding format-specific dependencies here: implement the public
:class:`TensorAdapter` contract and register it during application startup with
:func:`register_adapter`. Registered adapters are used for both movement and
tied-storage identity.

:class:`ResourceCache` manages cached backing stores with policy-driven
eviction, reference-counted leases, and transactional admission.
:class:`ModelCache` owns dependency leasing, LoRA attachment, and device
activation. Each model offloader rejects overlapping use. Custom
:class:`EvictionPolicy`
implementations can replace the default LRU behavior. See its docstring
for design notes.

Compatibility
-------------
- **``torch.compile`` is not supported** for managed modules.
- **Wrap before DDP/FSDP**, not after.
- **Coarse cache concurrency.** :class:`ResourceCache` serializes cache
  metadata and lease operations and releases its lock while caller code
  holds a lease. Model cache entries support one active use at a time; LoRA
  backing may be shared.
"""

from .gguf_adapter import GGUFWeight
from .lora import (
    LoRA,
    LoRAFactor,
    LoRAMode,
    LoRATransform,
    ScaledLoRAFactor,
)
from .merge import merge_lora
from .model_cache import ModelCache
from .model_offloader import ModelOffloader, ModelRuntimeInUseError
from .mps_weights import MpsWeights
from .pinned_component import PinnedComponent, PinnedComponentStore
from .protocols import (
    ResourceBinding,
    ResourceSpec,
    ResourceStore,
)
from .resource_cache import (
    CacheError,
    DuplicateResourceKeyError,
    EvictionCandidate,
    EvictionContext,
    EvictionPolicy,
    EvictionPolicyError,
    LRUEvictionPolicy,
    ResourceCache,
    ResourceCachedError,
    ResourceInfo,
    ResourceLeasedError,
    ResourceNotRegisteredError,
    ResourceTooLargeError,
)
from .resource_specs import LoRASpec, ModelSpec, ObjectSpec
from .stream_config import StreamConfig
from .streamed_component import StreamedComponent, StreamedComponentStore
from .tensor_adapter_registry import register_adapter
from .tensor_adapters import TensorAdapter

__all__ = [
    "CacheError",
    "DuplicateResourceKeyError",
    "EvictionCandidate",
    "EvictionContext",
    "EvictionPolicy",
    "EvictionPolicyError",
    "GGUFWeight",
    "LRUEvictionPolicy",
    "LoRA",
    "LoRAFactor",
    "LoRAMode",
    "LoRASpec",
    "LoRATransform",
    "ModelCache",
    "ModelOffloader",
    "ModelRuntimeInUseError",
    "ModelSpec",
    "MpsWeights",
    "ObjectSpec",
    "PinnedComponent",
    "PinnedComponentStore",
    "ResourceBinding",
    "ResourceCache",
    "ResourceCachedError",
    "ResourceInfo",
    "ResourceLeasedError",
    "ResourceNotRegisteredError",
    "ResourceSpec",
    "ResourceStore",
    "ResourceTooLargeError",
    "ScaledLoRAFactor",
    "StreamConfig",
    "StreamedComponent",
    "StreamedComponentStore",
    "TensorAdapter",
    "merge_lora",
    "register_adapter",
]
