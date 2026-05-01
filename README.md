# Memory

A model-agnostic GPU/CPU memory manager for PyTorch. Two pluggable
strategies for moving model weights between host and GPU, plus an LRU
cache that swaps multiple independent models in and out of GPU memory.

Self-contained, library-friendly: no dependencies beyond `torch` (plus
optional `optimum.quanto` for quantized models). Designed to be lifted
into its own package when a second consumer appears.

## What's in here

| Module | Role |
|---|---|
| `protocols.py` | `ModelStrategy` / `ModelStrategyComponent` plug-in contracts; `SlotOwnership` skip-filter type |
| `pinned_weights.py` | `PinnedWeights` — whole-model bulk pinned-CPU↔GPU strategy |
| `streamed_weights.py` | `StreamedWeights` — sharp per-block-list streaming primitive (component) |
| `model_offloader.py` | `ModelOffloader` — unified composite: block streaming + non-block pinning + trainable params + optional LoRA merge |
| `trainable_weights.py` | `TrainableWeights` — identity-preserving trainable parameter mover |
| `lora.py` | `LoRA`, `LoRATransform` — per-weight LoRA merge transform + factor pairing/validation |
| `pinned_buffer.py` | `PinnedParamBuffer` — per-tensor pinning primitive (handles quanto) |
| `tensor_adapters.py`, `quanto_adapter.py` | Tensor-type adapter registry and optional optimum-quanto support |
| `model_cache.py` | `ModelCache` — LRU pool over strategies with active-set leases |
| `pipeline_install.py` | LTX-specific integration shim; keep outside the generic core surface when publishing independently |

## Why use this

You have multiple PyTorch models that don't all fit on GPU
simultaneously, and you want to swap them in and out efficiently
across many calls. Re-loading from disk every call is too slow
(seconds per gigabyte). Keeping all models resident on GPU is too
expensive. `torch.cuda.empty_cache()` plus `.to("meta")` gets you the
basics but leaves significant performance on the table — pinned host
memory does CPU↔GPU DMA at full PCIe bandwidth (~30 GB/s vs.
~3 GB/s from disk), and a single LRU cache lets multiple models
share the same host-memory budget.

This library gives you:

1. **Strategies** that pin a model's frozen weights to host RAM and
   bulk-DMA them to GPU on demand.
2. **A cache** that holds multiple pinned models, evicts least-recently-
   used inactive entries when a new model needs room, and tracks active
   leases so you can't accidentally evict something you're using.
3. **A clean plug-in contract** so you can write your own strategy
   (disk-mmap, NVMe-paged, multi-GPU shard) and it slots in.

## When to use what

| Situation | Use |
|---|---|
| Model fits on GPU when active; want fast eviction between calls | **`PinnedWeights`** — bulk DMA, ~200 ms for 12 GB at PCIe Gen5 x16 |
| Model too big for GPU even when active | **`ModelOffloader`** — streams transformer blocks via forward hooks |
| Multiple models swap in/out across a script | Wrap each in a strategy, hand to **`ModelCache`** |

## Quick start: PinnedWeights

```python
import torch
from torch_offload import PinnedWeights

model = build_my_model()  # any nn.Module with frozen params
strategy = PinnedWeights(model, target_device=torch.device("cuda"))

# First use pays the pinning cost (clone + pin_memory).
# Subsequent uses skip pinning — bulk-DMA only.
with strategy as gpu_model:
    output = gpu_model(input_tensor)

with strategy as gpu_model:
    output = gpu_model(input_tensor_2)

del strategy, model  # drop refs to free pinned host memory
```

`PinnedWeights` mutates the model in place: every frozen
`nn.Parameter` slot gets repointed at a Parameter wrapping pinned CPU
storage. After construction, only access the model through the
strategy's context manager (or `activate()` / `deactivate()`).
**Drop the strategy and model references to release pinned host
memory** — there's no `close()`; resource cleanup is reference-drop
+ GC.

## Quick start: block streaming

For models too big to fit on GPU even when active. Streams transformer
blocks through a small GPU-resident window using forward-pre hooks
and a CUDA-stream-based async prefetcher.

```python
import torch
from torch_offload import ModelOffloader

# Constructor pins everything; cache_bytes is final immediately.
offloader = ModelOffloader(
    model,
    target_device=torch.device("cuda"),
    layers_attr="transformer_blocks",  # path to the nn.ModuleList
    blocks_to_swap=24,                 # offload N blocks; rest GPU-resident
    prefetch_count=2,
)

with offloader as gpu_model:
    output = gpu_model(input_tensor)

del offloader, model  # drop refs to free pinned host memory
```

Trainable parameters (e.g. LoRA adapters) move to GPU on activate
and back to CPU on deactivate via the bundled `TrainableWeights`
component — backward through them is unaffected by the offload.

### LoRA merge

`ModelOffloader` supports optional per-weight LoRA merging via
`set_loras()`. LoRA factors are attached as transforms on
`PinnedParamBuffer` objects and applied automatically after DMA —
both block-streamed and non-block weights get merged for free.
`set_loras()` first clears the currently attached transforms, then
validates and builds the replacement stack. If the replacement raises,
the offloader is left in base-only mode; this avoids briefly holding
both old and new pinned LoRA factors in host memory.

```python
from torch_offload import ModelOffloader, LoRA
from safetensors.torch import load_file

offloader = ModelOffloader(
    model,
    target_device=torch.device("cuda"),
    layers_attr="transformer_blocks",
    blocks_to_swap=24,
)

# Attach LoRAs (must be called while deactivated)
offloader.set_loras([
    (LoRA(state_dict=load_file("lora_a.safetensors")), 0.8),
    (LoRA(state_dict=load_file("lora_b.safetensors")), 0.5),
])

with offloader as gpu_model:
    output = gpu_model(input_tensor)

# Switch to different LoRAs or clear (base-only)
offloader.set_loras([])
```

Block reload from pristine pinned CPU storage automatically clears
the previous merge — no explicit unmerge step needed.

### Heterogeneous block lists

`layers_attr` accepts a list of dotted paths for models with
multiple kinds of blocks (e.g. Flux's `transformer_blocks` +
`single_transformer_blocks`). Each path becomes its own homogeneous
streaming group with its own slot pool — no per-load `cudaMalloc`
fallback:

```python
offloader = ModelOffloader(
    model,
    target_device=torch.device("cuda"),
    layers_attr=["transformer_blocks", "single_transformer_blocks"],
    blocks_to_swap=[8, 24],   # per-group; or pass a single int for both
    prefetch_count=[2, 4],
)
```

## Quick start: ModelCache

For multiple independent models swapping in and out of GPU.

```python
from torch_offload import ModelCache, ModelSpec, PinnedWeights

cache = ModelCache(max_cache_bytes=80 * 1024**3)

# Register specs (lazy — factory only runs on first acquire / cache miss)
cache.register(ModelSpec(
    key="text_encoder",
    estimated_cache_bytes=12 * 1024**3,
    factory=lambda: PinnedWeights(build_text_encoder(), device),
))
cache.register(ModelSpec(
    key="diffusion_model",
    estimated_cache_bytes=24 * 1024**3,
    factory=lambda: PinnedWeights(build_diffusion_model(), device),
))

# First use builds via factory; subsequent uses hit the cache.
with cache.use("text_encoder") as enc:
    embeddings = enc.encode(prompt)

with cache.use("diffusion_model") as t:
    latent = t(...)

# When budget pressure forces eviction, LRU inactive entries go first.
# Active entries (currently inside `cache.use(...)`) are never evicted.
```

You can also auto-register at acquire time:

```python
spec = ModelSpec(key="vae", estimated_cache_bytes=500*1024**2,
                 factory=lambda: PinnedWeights(build_vae(), device))
with cache.use(spec) as vae:  # registers if missing, then uses
    decoded = vae.decode(latent)
```

> **Anti-pattern:** the factory should build a fresh model each call,
> not capture an externally-held one. With `factory=lambda:
> PinnedWeights(my_kept_model, device)` the cache is no longer the
> sole owner of the model — eviction drops the strategy, but
> `my_kept_model` keeps the pinned slots alive. `used_cache_bytes`
> will lie about freed memory. Always have the factory build the
> model itself.

## Architecture

```
                       ┌──────────────────┐
                       │   ModelCache     │  LRU pool, active-set leases,
                       │                  │  transactional admission
                       └────────┬─────────┘
                                │ uses (via ModelStrategy protocol)
                                ▼
            ┌───────────────────┴────────────────────┐
            │                                        │
   ┌────────▼─────────┐                ┌─────────────▼──────────────┐
   │  PinnedWeights   │                │      ModelOffloader        │
   │  whole-model DMA │                │   (composes components)    │
   └────────┬─────────┘                └─────────────┬──────────────┘
            │                                        │
            │             ┌──────────────────────────┴──────────┐
            │             │  components (ordered):              │
            │             │  • PinnedWeights (non-block,        │
            │             │    skip_slots = streamers' slots)   │
            │             │  • TrainableWeights                 │
            │             │  • N × StreamedWeights              │
            │             │                                     │
            │             │  optional LoRA:                     │
            │             │  • LoRATransform on PinnedParamBuf  │
            │             └──────────────────────────┬──────────┘
            │                                        │
            └────────────────────┬───────────────────┘
                                 ▼
                       ┌──────────────────┐
                       │ PinnedParamBuffer│  per-tensor pinned-CPU storage
                       │  (quanto-aware)  │  + optional LoRA transform
                       └──────────────────┘
```

`ModelStrategy` is the protocol every top-level strategy implements:
`cache_bytes`, `model` (the wrapped module, stable across cycles),
`activate()`, `deactivate()`, plus context-manager dunders.
`ModelCache` only talks to this protocol; write a new strategy and
it slots in:

```python
from torch import nn

class MyStrategy:
    @property
    def cache_bytes(self) -> int: ...
    @property
    def model(self) -> nn.Module: ...
    def activate(self) -> None: ...
    def deactivate(self) -> None: ...
    def __enter__(self) -> nn.Module:
        self.activate()
        return self.model
    def __exit__(self, *exc) -> None:
        self.deactivate()
```

A narrower `ModelStrategyComponent` Protocol (just `cache_bytes` +
`activate` + `deactivate`, no `model`) describes pieces composable
inside a top-level strategy — `StreamedWeights`, `TrainableWeights`,
and a `PinnedWeights` used as a non-block sibling all satisfy it.

## Strategy lifecycle

Uniform across all strategies — pinning happens in `__init__` so
`cache_bytes` is final at admission time:

```
constructed → activate ↔ deactivate → drop refs
```

`activate()` makes the model usable for compute. `deactivate()`
releases transient GPU resources (the `cache_bytes` worth of pinned
storage stays held in module slots, ready for fast re-activation).
**There is no `close()`.** To release pinned host memory, drop the
strategy reference (and the model reference if you don't need it
anymore). Python's refcount-based GC frees pinned tensors
immediately. Strategies release what they own; ownership of the
user's model is the user's concern.

**Failure semantics.** If `activate()` raises midway, the strategy
is poisoned — drop the strategy reference and rebuild. Don't retry
`activate()` on a failed strategy. This is a low-level library; we
don't guard against caller misuse.

## Compatibility

- **`torch.compile` is not supported** for managed modules. Both
  strategies swap parameter slots (`module._parameters[leaf] = new_param`)
  on every activate/deactivate, and `StreamedWeights` registers
  forward-pre hooks that mutate slots on every block call. Both
  invalidate the tensor-identity assumptions `torch.compile` makes
  about its trace.
- **Wrap before DDP/FSDP**, not after. Those wrappers manage parameter
  storage themselves and conflict with the slot-swap pattern.
- **Single-thread / sequential.** No internal locking; concurrent use
  on the same strategy or cache is undefined behavior.
- **Buffer mutations during forward are discarded** on `deactivate()`.
  Suitable for inference of stateless modules; not suitable for models
  that need persistent buffer state across calls (BatchNorm running
  stats updated in training mode, RNN/SSM hidden state, KV cache).

## Tied weights

Both strategies handle the standard `tie_weights()` pattern (one
`Parameter` referenced under multiple names) plus the rarer case of
distinct quanto wrappers around shared inner `_data` storage.

`ModelOffloader` rejects (at construction) tied weights that
span streamed regions — block↔block, block↔non-block, or mixed
trainable/frozen across regions. Slot-local block streaming can't
preserve cross-region tying. Use whole-model `PinnedWeights` for
those models instead.

## Quanto support

Quanto-quantized models (`optimum.quanto.WeightQBytesTensor`) are
handled correctly by both strategies. `PinnedParamBuffer` decomposes
the wrapper into its inner `_data` (int8/fp8) and `_scale` (fp16/fp32)
tensors, pins each, and reconstructs the quanto wrapper around the GPU
storage on activation.

A naive `param.data.clone()` on a quanto tensor silently
*dequantizes* it via the dispatch fallback — the explicit decomposition
is required for correctness.

## Failure modes

The cache and strategies surface failures as typed exceptions rather
than silent corruption.

| Exception | When |
|---|---|
| `ModelTooLargeError` | Cache miss can't fit even after evicting all inactive entries (active entries blocking) |
| `ActivationError` | Strategy's `activate()` raised — the cache discards the entry; next acquire rebuilds |
| `ModelInUseError` | `evict()` / `clear()` / `unregister()` called while entry is active |
| `DuplicateModelKeyError` | `register()` called for an existing key without `replace=True` |
| `ModelNotRegisteredError` | `use(str)` called for an unknown key |

## Observability

```python
snap = cache.snapshot()
snap.used_cache_bytes        # current pinned-host bytes
snap.cached_keys_lru_to_mru  # tuple of currently-cached keys
snap.active_refcounts        # tuple of (key, refcount) for active entries
snap.stats.hits              # cache hit count
snap.stats.evictions         # total evictions
snap.stats.peak_cache_bytes  # high-water mark
```

`ModelCacheSnapshot`, `ModelCacheStats`, and `ModelInfo` are direct
imports from `torch_offload.model_cache` (not re-exported at the
package level — they're observability types, not the typical
acquire/use path).
