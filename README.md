# Memory

A model-agnostic GPU/CPU memory manager for PyTorch. Two pluggable
strategies for moving model weights between host and GPU, plus a
policy-driven cache that swaps multiple independent models in and out
of GPU memory.

Self-contained, library-friendly: no dependencies beyond `torch` (plus
optional `optimum.quanto` and `gguf` for quantized models). Designed
to be lifted into its own package when a second consumer appears.

## What's in here

| Module | Role |
|---|---|
| `protocols.py` | `CachedResource` (generic), `ModelStrategy` / `ModelStrategyComponent` plug-in contracts; `SlotOwnership` skip-filter type |
| `pinned_weights.py` | `PinnedWeights` — whole-model bulk pinned-CPU↔GPU strategy |
| `streamed_weights.py` | `StreamedWeights` — sharp per-block-list streaming primitive (component) |
| `model_offloader.py` | `ModelOffloader` — unified composite: block streaming + non-block pinning + trainable params + optional LoRA merge |
| `trainable_weights.py` | `TrainableWeights` — identity-preserving trainable parameter mover |
| `lora.py` | `LoRA`, `LoRATransform`, `LoRARouteHandle` — pinned factor storage + merge / routed-hook application |
| `merge.py` | `merge_lora()` — permanent in-place LoRA merge into base weights (alternative to `set_loras`) |
| `pinned_buffer.py` | `PinnedParamBuffer` — per-tensor pinning primitive (handles quanto + GGUF via adapters) |
| `tensor_adapters.py`, `quanto_adapter.py`, `gguf_adapter.py`, `gguf_dequant.py` | Tensor-type adapter registry and optional optimum-quanto / gguf support |
| `_quanto.py` | Internal: optimum-quanto optional-import + layout validation; consumed by `quanto_adapter.py` and `merge.py` |
| `slots.py` | Slot-resolution helpers: `iter_param_slots`, `iter_buffer_slots`, `assert_frozen`, `canonical_param_name`, dotted-path walkers |
| `model_cache.py` | `ModelCache` — policy-driven pool over `CachedResource` instances with active-set leases |

## Why use this

You have multiple PyTorch models that don't all fit on GPU
simultaneously, and you want to swap them in and out efficiently
across many calls. Re-loading from disk every call is too slow
(seconds per gigabyte). Keeping all models resident on GPU is too
expensive. `torch.cuda.empty_cache()` plus `.to("meta")` gets you the
basics but leaves significant performance on the table — pinned host
memory does CPU↔GPU DMA at full PCIe bandwidth (~30 GB/s vs.
~3 GB/s from disk), and a shared cache lets multiple models use the
same host-memory budget.

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
| Model too big for a CUDA GPU even when active | **`ModelOffloader`** — streams transformer blocks via forward hooks |
| Multiple models swap in/out across a script | Wrap each in a strategy, hand to **`ModelCache`** |

## Quick start: PinnedWeights

```python
import torch
from torch_offload import PinnedWeights

model = build_my_model()  # any nn.Module with frozen params
strategy = PinnedWeights(model)
device = torch.device("cuda")

# Construction pays the pinning cost (clone + pin_memory).
# Each use is bulk-DMA only.
with strategy.use(device) as gpu_model:
    output = gpu_model(input_tensor)

with strategy.use(device) as gpu_model:
    output = gpu_model(input_tensor_2)

del strategy, model  # drop refs to free pinned host memory
```

`PinnedWeights` mutates the model in place: every frozen
`nn.Parameter` slot gets repointed at a Parameter wrapping pinned CPU
storage. After construction, only access the model through the
strategy's `use(device)` context manager (or `activate(device)` /
`deactivate()`).
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
    layers_attr="transformer_blocks",  # path to the nn.ModuleList
    blocks_to_swap=24,                 # offload N blocks; rest GPU-resident
    prefetch_count=2,
)
device = torch.device("cuda")

with offloader.use(device) as gpu_model:
    output = gpu_model(input_tensor)

del offloader, model  # drop refs to free pinned host memory
```

`ModelOffloader` only streams on CUDA. Activating the base offloader on
`cpu` is a pass-through over the already-installed pinned CPU storage:
no slot pool, no streaming hooks, no weight copies.
`set_loras(..., mode="merge")` is CUDA-only; use routed LoRA mode for
CPU activation. Routed LoRA still installs forward hooks and materializes
LoRA factors on the activation device.

By default, trainable parameters (e.g. LoRA adapters) move to GPU on
CUDA activation and back to CPU on deactivate via the bundled
`TrainableWeights` component. On CPU activation they stay in the
host-backed module slots. This preserves ordinary PyTorch optimizer
behavior: `optimizer.step()`, gradient accumulation, AMP unscale, and
grad clipping see normal trainables on the activation device while the
offloader is active.

To reduce trainable-weight residency during training, opt into
streaming in-block trainable weights:

```python
offloader = ModelOffloader(
    model,
    layers_attr="transformer_blocks",
    blocks_to_swap=24,
    stream_trainable_weights=True,
)
```

During CUDA activation in this mode, only the trainable parameter
`.data` streams. It is GPU-resident while its block is resident, plus
during the optimizer update. CPU activation remains pass-through.
Gradients are not streamed; PyTorch owns `param.grad` normally.

### LoRA merge

`ModelOffloader` supports optional per-weight LoRA merging via
`set_loras()`. LoRA requests are applied during activation; merge mode
installs activation-scoped post-copy hooks for matched targets. Each
hook runs immediately after the owning component copies the base weight
from pinned CPU storage to GPU, so both block-streamed and non-block
weights use the same merge path. `PinnedParamBuffer` remains a storage
primitive; it does not own LoRA-specific behavior.
`set_loras()` records the replacement request while the offloader is
inactive. Target matching and mode-specific compatibility are validated
during activation, so LoRA application follows the same runtime boundary
as device selection.

```python
import torch
from torch_offload import ModelOffloader, LoRA
from safetensors.torch import load_file

offloader = ModelOffloader(
    model,
    layers_attr="transformer_blocks",
    blocks_to_swap=24,
    # Default: stream_trainable_weights=False
)
device = torch.device("cuda")

# Request LoRAs for the next activation (must be called while deactivated)
offloader.set_loras([
    (LoRA(state_dict=load_file("lora_a.safetensors")), 0.8),
    (LoRA(state_dict=load_file("lora_b.safetensors")), 0.5),
])

with offloader.use(device) as gpu_model:
    output = gpu_model(input_tensor)

# Switch to different LoRAs or clear (base-only)
offloader.set_loras([])
```

Block reload from pristine pinned CPU storage automatically clears
the previous merge — no explicit unmerge step needed.

`set_loras` accepts `mode="routed"` as an alternative to the default
`mode="merge"`. Routed mode installs a forward hook on each matched
`nn.Linear` parent — `y = base(x) + alpha * B * A * x` — instead of merging
into the base weight. Use it when:

- The base weight is quantized (quanto): `mode="merge"` rejects on activation
  subclassed wrappers because in-place `addmm_` silently drops the
  update on them; `mode="routed"` works because it doesn't touch
  the base.
- You want to switch LoRAs frequently without re-streaming the
  underlying base weight.

Routed mode is restricted to `nn.Linear` parents and rejects tied
weights (the hook would only fire on one alias).

For a one-shot **permanent** merge — bake the LoRA into the model
weights and discard the LoRA — use `merge_lora`:

```python
from torch_offload import merge_lora, LoRA

merge_lora(model, [(LoRA(state_dict=load_file("lora.safetensors")), 0.8)])
```

This dequantizes-and-requantizes for quanto bases (lossy but
standard) and uses an in-place `addmm_` for fp/bf bases. Unlike
`set_loras`, this is not reversible.

### Heterogeneous block lists

`layers_attr` accepts a list of dotted paths for models with
multiple kinds of blocks (e.g. Flux's `transformer_blocks` +
`single_transformer_blocks`). Each path becomes its own streaming
group with its own slot pool. Blocks within a group must share the
same parameter layout (names/shapes/dtypes/quant-metadata) — split
heterogeneous block lists into separate `layers_attr` entries:

```python
offloader = ModelOffloader(
    model,
    layers_attr=["transformer_blocks", "single_transformer_blocks"],
    blocks_to_swap=[8, 24],   # per-group; or pass a single int for both
    prefetch_count=[2, 4],
)
```

### Training streamed blocks

Training through a streamed block **requires activation checkpointing
on each block** — wrap call sites in
`torch.utils.checkpoint.checkpoint`, or call
`model.gradient_checkpointing_enable()` on a HuggingFace model.
Without it, `loss.backward()` raises:

```
RuntimeError: one of the variables needed for gradient computation
has been modified by an inplace operation
```

The reason is autograd's saved-tensor mechanism. A `Linear` saves a
reference to its weight tensor at forward time and records the
tensor's version counter. Streaming is a sequence of in-place `copy_`
writes into a fixed pool of GPU slot tensors — every block load
bumps the slot tensor's version, so by the time backward arrives at
an earlier block, the slot has been overwritten and the version
mismatch raises.

Activation checkpointing sidesteps this. With checkpointing, the
block's internal forward runs under `no_grad` — no internal tensors
are saved for backward. When backward arrives, PyTorch re-runs the
block's forward with grad enabled, building a fresh autograd graph
whose saved references only live within that one block's
recompute-then-backward window. Slot reuse outside that window is
safe because no autograd graph spans across reuses.

```python
import torch
from torch_offload import ModelOffloader

offloader = ModelOffloader(
    model,
    layers_attr="transformer_blocks",
    blocks_to_swap=24,
)
device = torch.device("cuda")

model.gradient_checkpointing_enable()  # required for training
model.train()

with offloader.use(device) as gpu_model:
    for batch in loader:
        loss = gpu_model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

`ModelOffloader.activate()` emits a one-time warning if
`model.training=True` with trainable params present and no
HuggingFace `gradient_checkpointing` flag is detected on the streamed
blocks. With `stream_trainable_weights=True`, the same missing check is
a hard error because trainable weight streaming can silently corrupt
gradients without checkpointing. Manual `checkpoint(...)` wrapping at
call sites is invisible from the module tree; pass
`skip_checkpointing_check=True` only after verifying every streamed
training block is checkpointed.

For `stream_trainable_weights=True`, wrap the optimizer update
so streamed trainable weights are materialized on GPU while a normal
PyTorch optimizer mutates it:

```python
with offloader.use(device) as gpu_model:
    for batch in loader:
        loss = gpu_model(**batch).loss
        loss.backward()

        with offloader.optimizer_step():
            optimizer.step()

        optimizer.zero_grad()
```

This boundary is not optimizer-specific. It temporarily materializes
the streamed trainable `.data` tensors on GPU, runs whatever
`optimizer.step()` does, copies updated data back to pinned CPU, and
leaves gradients on GPU.

## Quick start: ModelCache

For multiple independent models swapping in and out of GPU.

```python
from torch_offload import ModelCache, ModelSpec, PinnedWeights

cache = ModelCache(max_cache_bytes=80 * 1024**3)
device = "cuda:0"

# Register specs (lazy — factory only runs on first acquire / cache miss)
cache.register(ModelSpec(
    key="text_encoder",
    estimated_cache_bytes=12 * 1024**3,
    factory=lambda: PinnedWeights(build_text_encoder()),
))
cache.register(ModelSpec(
    key="diffusion_model",
    estimated_cache_bytes=24 * 1024**3,
    factory=lambda: PinnedWeights(build_diffusion_model()),
))

# First use builds via factory; subsequent uses hit the cache.
with cache.use("text_encoder", device=device) as enc:
    embeddings = enc.encode(prompt)

with cache.use("diffusion_model", device=device) as t:
    latent = t(...)

# When budget pressure forces eviction, the eviction policy chooses
# from inactive cached entries. The default policy is LRU.
# Active entries (currently inside `cache.use(...)`) are never evicted.
# Re-entrant use of the same key must use the same device; simultaneous
# activation of one cached strategy on multiple devices is rejected.
# CUDA devices use a simple placement guard: one active model key per
# cache-visible CUDA device passed to `cache.use(..., device=...)`.
```

You can also auto-register at acquire time:

```python
spec = ModelSpec(key="vae", estimated_cache_bytes=500*1024**2,
                 factory=lambda: PinnedWeights(build_vae()))
with cache.use(spec, device=device) as vae:  # registers if missing, then uses
    decoded = vae.decode(latent)
```

> **Anti-pattern:** the factory should build a fresh model each call,
> not capture an externally-held one. With `factory=lambda:
> PinnedWeights(my_kept_model)` the cache is no longer the
> sole owner of the model — eviction drops the strategy, but
> `my_kept_model` keeps the pinned slots alive. `used_cache_bytes`
> will lie about freed memory. Always have the factory build the
> model itself.

`ModelCache` accepts custom `EvictionPolicy` and `PlacementPolicy`
implementations. Defaults are `LRUEvictionPolicy` for inactive host
cache eviction and `OneModelPerCudaDevicePolicy` for CUDA placement.
The cache builds the eviction candidate set and byte context, then asks
the eviction policy to choose victims; `ModelCache` still owns
validation, accounting, activation, rollback, and release. Policies are
called under the cache lock. `choose_victims()` must return unique keys
from `context.candidates` and enough bytes to satisfy
`context.bytes_to_free`; otherwise `ModelCache` raises
`EvictionPolicyError` without evicting anything.

## Architecture

```
                       ┌──────────────────┐
                       │   ModelCache     │  policy eviction, leases,
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
            │             │  • post-copy hooks for merge mode   │
            │             └──────────────────────────┬──────────┘
            │                                        │
            └────────────────────┬───────────────────┘
                                 ▼
                       ┌──────────────────┐
                       │ PinnedParamBuffer│  per-tensor pinned-CPU storage
                       │  (quanto-aware)  │  via tensor adapters
                       └──────────────────┘
```

`ModelStrategy` is the protocol every top-level strategy implements:
`cache_bytes`, `model` (the wrapped module, stable across cycles),
`activate(device=None)`, and `deactivate()`. Device-aware package
strategies also expose `use(device)` for direct exception-safe use.
`ModelCache` only talks to this protocol; write a new strategy and
it slots in:

```python
from torch import nn

class MyStrategy:
    @property
    def cache_bytes(self) -> int: ...
    @property
    def model(self) -> nn.Module: ...
    def activate(self, device=None) -> None: ...
    def deactivate(self) -> None: ...
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

`activate(device=...)` makes the model usable for compute on the
requested device. `PinnedWeights`, `ModelOffloader`, `StreamedWeights`,
and `TrainableWeights` require an explicit device. CUDA activation uses
the streaming/DMA path where applicable; CPU activation is pass-through
over pinned host-backed storage.
`deactivate()` releases transient device resources (the `cache_bytes`
worth of pinned storage stays held in module slots, ready for fast
re-activation).

Construction optimizes peak host memory. Pinning clones managed tensors
into pinned CPU storage. For plain `torch.Tensor` parameters, the source
`Parameter.data` may be immediately repointed at the pinned clone as soon
as that buffer is created. This releases the original pageable CPU
storage early and avoids temporarily holding both pageable and pinned
copies of large weights. It is a clone-to-pinned plus storage swap, not
true in-place pinning. Tensor subclasses such as quanto/GGUF do not use
this `.data` swap when it would lose wrapper state.

**There is no `close()`.** To release pinned host memory, drop the
strategy reference (and the model reference if you don't need it
anymore). Python's refcount-based GC frees pinned tensors
immediately. Strategies release what they own; ownership of the
user's model is the user's concern.

**Failure semantics.** If construction raises after pinning has started,
the model may already be partially repointed to pinned storage. Treat the
partially constructed strategy/model as unrecoverable: drop those
references and rebuild from a fresh model instance. If `activate()`
raises midway, the strategy may contain partial device state; don't retry
`activate()` on that strategy. Drop the strategy reference and rebuild.
This is a low-level library; we don't guard against caller misuse.

## Compatibility

- **`torch.compile` is not supported** for managed modules. Both
  strategies swap parameter slots (`module._parameters[leaf] = new_param`)
  on every activate/deactivate, and `StreamedWeights` registers
  forward-pre hooks that mutate slots on every block call. Both
  invalidate the tensor-identity assumptions `torch.compile` makes
  about its trace.
- **Wrap before DDP/FSDP**, not after. Those wrappers manage parameter
  storage themselves and conflict with the slot-swap pattern.
- **Coarse cache concurrency.** `ModelCache` protects cache metadata,
  admission, leases, and the simple one-key-per-CUDA-device
  placement table with an internal lock. Factory/build, activation, and
  deactivation are serialized, but the lock is released while caller
  code runs inside `cache.use(...)`. The cache does not make a yielded
  model object safe for concurrent same-key execution.
- **Buffer mutations during CUDA activation are discarded** on
  `deactivate()`. CPU activation is pass-through over host-backed
  buffers, so CPU buffer mutations behave like ordinary module
  mutations. Suitable for inference of stateless modules; not suitable
  for models that need persistent buffer state across calls (BatchNorm
  running stats updated in training mode, RNN/SSM hidden state, KV
  cache).
- **Training requires activation checkpointing** on every streamed
  block (`model.gradient_checkpointing_enable()` for HF models, or
  manual `torch.utils.checkpoint.checkpoint` wrapping). Without it,
  `loss.backward()` raises an in-place modification error from
  autograd's saved-tensor check. See
  [Training streamed blocks](#training-streamed-blocks).

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

LoRA on quanto bases: merge mode rejects quanto targets on activation
(in-place `addmm_` on a `WeightQBytesTensor` returns
success but silently leaves `_data` untouched). Use
`set_loras(mode="routed")` for inference-time application, or
`merge_lora()` for a permanent dequant -> addmm -> requant bake-in.

## Failure modes

The cache and strategies surface failures as typed exceptions rather
than silent corruption.

| Exception | When |
|---|---|
| `ModelTooLargeError` | Cache miss can't fit even after evicting all inactive entries. Exposes `required`, `used`, and `limit`. |
| `EvictionPolicyError` | Custom eviction policy returned duplicate/non-candidate victims or too few bytes |
| `ActivationError` | Strategy's `activate()` raised — the cache discards the entry; next acquire rebuilds |
| `GpuDeviceOccupiedError` | A CUDA device already has a different active model key under the simple placement guard |
| `ModelInUseError` | `evict()` / `clear()` / `unregister()` called while entry is active |
| `DuplicateModelKeyError` | `register()` called for an existing key without `replace=True` |
| `ModelNotRegisteredError` | `use(str)` called for an unknown key |

## State Inspection

Use `cache.used_cache_bytes` and `cache.available_cache_bytes` for
current cache accounting. `available_cache_bytes` is signed; negative
means a strategy reported growth after admission and the cache is
currently over budget. Use `cache.info(key)` for per-key state when
needed.
