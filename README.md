# Memory

A model-agnostic GPU/CPU memory manager for PyTorch. Two pluggable
strategies for moving model weights between host and GPU, plus a
policy-driven cache that swaps multiple independent models in and out
of GPU memory.

Self-contained, library-friendly: no dependencies beyond `torch` (plus
optional `optimum.quanto`, `gguf`, and `torchao` for quantized models). Designed
to be lifted into its own package when a second consumer appears.

## What's in here

| Module | Role |
|---|---|
| `protocols.py` | `ResourceStore`, `ResourceBinding`, `ModelStrategy` / `ModelStrategyComponent` plug-in contracts |
| `model_offloader.py` | `ModelOffloader` — whole-model bulk pinned-CPU↔GPU or streamed block offload strategy |
| `pinned_component.py` | `PinnedComponent`, `PinnedComponentStore` — reusable pinned backing storage plus lifecycle-only pinned component used by `ModelOffloader` |
| `streamed_component.py` | `StreamedComponent`, `StreamedComponentStore` — reusable streamed backing storage plus sharp per-block-list streaming component |
| `lora.py` | `LoRA`, `LoRATransform`, `LoRARouteHandle` — pinned factor storage + merge / routed-hook application |
| `merge.py` | `merge_lora()` — permanent in-place LoRA merge into base weights (alternative to `set_loras`) |
| `pinned_param.py` | `PinnedParam` — per-parameter pinning primitive (handles quanto, GGUF, and TorchAO NVFP4 via adapters) |
| `pinned_module.py` | Internal name-keyed pinned module storage plus concrete module bindings |
| `tensor_adapters.py`, `quanto_adapter.py`, `gguf_adapter.py`, `nvfp4_adapter.py`, `gguf_dequant.py` | Tensor adapter contracts/implementations and optional optimum-quanto / gguf / torchao support |
| `tensor_adapter_registry.py` | Internal adapter dispatch and tensor-identity helpers |
| `module_names.py` | Internal name traversal and mutation helpers |
| `_quanto.py` | Internal: optimum-quanto optional-import + layout validation; consumed by `quanto_adapter.py` and `merge.py` |
| `_torchao_nvfp4.py` | Internal: TorchAO NVFP4 optional-import + layout validation; consumed by `nvfp4_adapter.py` |
| `model_cache.py` | `ModelCache` — policy-driven pool over cached stores with per-use bindings |

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

1. **Strategies** that pin a model's weights to host RAM and
   bulk-DMA them to GPU on demand.
2. **A cache** that holds multiple pinned models, evicts least-recently-
   used inactive entries when a new model needs room, and tracks active
   leases so you can't accidentally evict something you're using.
3. **A clean plug-in contract** so you can write your own store/binding
   resource (disk-mmap, NVMe-paged, multi-GPU shard) and it fits in.

## When to use what

| Situation | Use |
|---|---|
| Model fits on GPU when active; want fast eviction between calls | **`ModelOffloaderStore.from_module(model).bind(model)`** — bulk DMA, ~200 ms for 12 GB at PCIe Gen5 x16 |
| Model too big for a CUDA GPU even when active | **`ModelOffloaderStore.from_module(model, layers_attr=..., blocks_to_swap=...).bind(model)`** — streams transformer blocks via forward hooks |
| Multiple models swap in/out across a script | Register model factories with **`ModelCache`** via **`ModelSpec`** |

## Quick start: whole-model offload

```python
import torch
from torch_offload import ModelOffloaderStore

model = build_my_model()  # any nn.Module
store = ModelOffloaderStore.from_module(model)
strategy = store.bind(model)
device = torch.device("cuda")

# Store construction pays the pinning cost (clone + pin_memory).
# Each use is bulk-DMA only.
with strategy.use(device) as gpu_model:
    output = gpu_model(input_tensor)

with strategy.use(device) as gpu_model:
    output = gpu_model(input_tensor_2)

del strategy, model  # drop refs to free pinned host memory
```

`ModelOffloader` mutates the model in place: frozen `nn.Parameter`
registry entries get repointed at Parameters wrapping pinned CPU storage,
trainable Parameter objects keep their identity and point their
`.data` at pinned CPU storage, and buffers are replaced with pinned
copies. After construction, only access the model through the strategy's
`use(device)` context manager (or `activate(device)` / `deactivate()`).
For CUDA training, wrap `optimizer.step()` in
`strategy.optimizer_step()` so trainable GPU updates are copied back to
the pinned CPU cache before deactivation.
**Drop the strategy and model references to release pinned host
memory** — there's no `close()`; resource cleanup is reference-drop
+ GC.

## Quick start: block streaming

For models too big to fit on GPU even when active. Streams transformer
blocks through a small GPU-resident window using forward-pre hooks
and a CUDA-stream-based async prefetcher.

```python
import torch
from torch_offload import ModelOffloaderStore

# Store construction pins everything; cache_bytes is final immediately.
store = ModelOffloaderStore.from_module(
    model,
    layers_attr="transformer_blocks",  # path to the nn.ModuleList
    blocks_to_swap=24,                 # offload N blocks; rest GPU-resident
    prefetch_count=2,
)
offload = store.bind(model)
device = torch.device("cuda")

with offload.use(device) as gpu_model:
    output = gpu_model(input_tensor)

del offload, model  # drop refs to free pinned host memory
```

`ModelOffloader` only streams on CUDA. Activating the strategy on
`cpu` is a pass-through over the already-installed pinned CPU storage:
no target pool, no streaming hooks, no weight copies.
`set_loras(..., mode="merge")` is CUDA-only; use routed LoRA mode for
CPU activation. Routed LoRA still installs forward hooks and materializes
LoRA factors on the activation device.

By default, trainable parameters (e.g. LoRA adapters) are managed by
the composed `PinnedComponent`: they move to GPU on CUDA activation and
back to pinned CPU storage on deactivate. On CPU activation they stay in
the host-backed module state. Wrap CUDA optimizer updates in
`offload.optimizer_step()` so updated trainable bytes are copied back
to pinned CPU storage before deactivation.

To reduce trainable-weight residency during training, opt into
streaming in-block trainable weights:

```python
store = ModelOffloaderStore.from_module(
    model,
    layers_attr="transformer_blocks",
    blocks_to_swap=24,
    stream_trainable_weights=True,
)
offload = store.bind(model)
```

During CUDA activation in this mode, only the trainable parameter
`.data` streams. It is GPU-resident while its block is resident, plus
during the optimizer update. CPU activation remains pass-through.
Gradients are not streamed; PyTorch owns `param.grad` normally.

### LoRA merge

`ModelOffloader` supports optional per-weight LoRA merging via
`set_loras()`. LoRA requests are applied during activation; merge mode
installs activation-scoped post-copy hooks for canonical managed
parameter targets. Unknown targets raise during activation. PEFT
`.base_layer.` model parameter paths are canonicalized for lookup, so
LoRA target keys should use the logical form like
`blocks.0.attn.weight`. Each hook runs immediately after the owning
component copies the base weight from pinned CPU storage to GPU, so both
block-streamed and non-block weights use the same merge path. Merge
compatibility is adapter-owned: plain dense tensors opt into in-place
`addmm_`; structured quantized wrappers can opt into
dequantize/requantize plus `copy_into` updates. Use routed mode for
formats that do not expose either merge capability but still provide a
compatible logical Linear weight shape and compute dtype. `PinnedParam`
remains a storage primitive; LoRA merge mode asks the selected adapter
for the required update capability.

`set_loras()` records the replacement request while the offload is
inactive. Target lookup is resolved during activation; target
compatibility can be preflighted with `LoRATransform.validate_target()`
or validated when the merge hook applies.

```python
import torch
from torch_offload import ModelOffloaderStore, LoRA
from safetensors.torch import load_file

store = ModelOffloaderStore.from_module(
    model,
    layers_attr="transformer_blocks",
    blocks_to_swap=24,
    # Default: stream_trainable_weights=False
)
offload = store.bind(model)
device = torch.device("cuda")

# Request LoRAs for the next activation (must be called while deactivated)
lora_a = LoRA(state_dict=load_file("lora_a.safetensors"))
lora_b = LoRA(state_dict=load_file("lora_b.safetensors"))
offload.set_loras([lora_a, lora_b], strengths=[0.8, 0.5])

with offload.use(device) as gpu_model:
    output = gpu_model(input_tensor)

# Switch to different LoRAs or clear (base-only)
offload.set_loras([])
```

Block reload from pristine pinned CPU storage automatically clears
the previous merge — no explicit unmerge step needed.

`set_loras` accepts `mode="routed"` as an alternative to the default
`mode="merge"`. Routed mode installs a forward hook on each matched
`nn.Linear` parent — `y = base(x) + alpha * B * A * x` — instead of merging
into the base weight. Use it when:

- The base weight is quantized or otherwise structured, but still exposes
  a logical `nn.Linear` weight shape and compute dtype, and its adapter
  does not support merge updates. `mode="routed"` works because it
  doesn't touch the base.
- You want to switch LoRAs frequently without re-streaming the
  underlying base weight.

Routed mode is restricted to `nn.Linear` parents and rejects tied
weights (the hook would only fire on one alias). Packed formats whose
parameter shape differs from the logical matmul weight need a per-format
route layer.

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
group with its own target pool. Blocks within a group must share the
same parameter layout (names/shapes/dtypes/quant-metadata) — split
heterogeneous block lists into separate `layers_attr` entries:

```python
store = ModelOffloaderStore.from_module(
    model,
    layers_attr=["transformer_blocks", "single_transformer_blocks"],
    blocks_to_swap=[8, 24],   # per-group; or pass a single int for both
    prefetch_count=[2, 4],
)
offload = store.bind(model)
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
writes into a fixed pool of GPU target tensors — every block load
bumps the target tensor's version, so by the time backward arrives at
an earlier block, the target has been overwritten and the version
mismatch raises.

Activation checkpointing sidesteps this. With checkpointing, the
block's internal forward runs under `no_grad` — no internal tensors
are saved for backward. When backward arrives, PyTorch re-runs the
block's forward with grad enabled, building a fresh autograd graph
whose saved references only live within that one block's
recompute-then-backward window. Target reuse outside that window is
safe because no autograd graph spans across reuses.

```python
import torch
from torch_offload import ModelOffloaderStore

store = ModelOffloaderStore.from_module(
    model,
    layers_attr="transformer_blocks",
    blocks_to_swap=24,
)
offload = store.bind(model)
device = torch.device("cuda")

model.gradient_checkpointing_enable()  # required for training
model.train()

with offload.use(device) as gpu_model:
    for batch in loader:
        loss = gpu_model(**batch).loss
        loss.backward()
        with offload.optimizer_step():
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

Wrap CUDA optimizer updates so managed trainable weights are synced back
to pinned CPU storage. With `stream_trainable_weights=True`, this also
materializes streamed trainable weights on GPU while a normal PyTorch
optimizer mutates them:

```python
with offload.use(device) as gpu_model:
    for batch in loader:
        loss = gpu_model(**batch).loss
        loss.backward()

        with offload.optimizer_step():
            optimizer.step()

        optimizer.zero_grad()
```

This boundary is not optimizer-specific. It runs whatever
`optimizer.step()` does, copies updated trainable data back to pinned
CPU storage, and leaves gradients on GPU.

## Quick start: ModelCache

For multiple independent models swapping in and out of GPU.

```python
from torch_offload import ModelCache, ModelSpec

cache = ModelCache(max_cache_bytes=80 * 1024**3)
device = "cuda:0"

# Register specs. The factory builds real weights once to create the
# cached store. Frozen models use allocation-light meta skeletons for
# bindings; trainable models reuse the primary factory-created model.
cache.register(ModelSpec(
    key="text_encoder",
    estimated_cache_bytes=12 * 1024**3,
    factory=build_text_encoder,
))
cache.register(ModelSpec(
    key="diffusion_model",
    estimated_cache_bytes=24 * 1024**3,
    factory=build_diffusion_model,
    layers_attr="transformer_blocks",
    blocks_to_swap=24,
))

# First use builds the store; subsequent uses reuse it. Frozen models
# get fresh bindings; trainable models reuse the primary model so
# optimizer parameter identity stays stable.
with cache.use("text_encoder", device=device) as enc:
    embeddings = enc.encode(prompt)

with cache.use("diffusion_model", device=device) as t:
    latent = t(...)

# When budget pressure forces eviction, the eviction policy chooses
# from inactive cached entries. The default policy is LRU.
# Active entries (currently inside `cache.use(...)`) are never evicted.
# Nested use of the same frozen key creates a second binding from the
# same cached store. Trainable same-key nesting is rejected. Caller code
# owns VRAM planning.
```

You can also auto-register at acquire time:

```python
spec = ModelSpec(key="vae", estimated_cache_bytes=500*1024**2,
                 factory=build_vae)
with cache.use(spec, device=device) as vae:  # registers if missing, then uses
    decoded = vae.decode(latent)
```

LoRA adapters can be cached as resources and applied to a model use
before activation:

```python
from torch_offload import LoRASpec
from safetensors.torch import load_file

lora = LoRASpec(
    key="style-lora",
    estimated_cache_bytes=512 * 1024**2,
    factory=lambda: load_file("style.safetensors"),
)

with cache.use(
    "diffusion_model",
    device=device,
    loras=[lora],
    lora_strengths=[0.8],
    lora_mode="routed",
) as t:
    latent = t(...)
```

> **Anti-pattern:** the factory should build a fresh model each call,
> not capture an externally-held one. With `factory=lambda:
> my_kept_model` the cache is no longer the sole owner of the model.
> Always have the factory build the model itself.

`ModelCache` accepts custom `EvictionPolicy` implementations. The
default is `LRUEvictionPolicy` for inactive host-cache eviction. The
cache builds the eviction candidate set and byte context, then asks the
eviction policy to choose victims; `ModelCache` still owns validation,
accounting, binding, activation, rollback, and release. Policies are called under
the cache lock. `choose_victims()` must return unique keys from
`context.candidates` and enough bytes to satisfy
`context.bytes_to_free`; otherwise `ModelCache` raises
`EvictionPolicyError` without evicting anything.

## Architecture

```
                       ┌──────────────────┐
                       │   ModelCache     │  policy eviction, bindings,
                       │                  │  transactional admission
                       └────────┬─────────┘
                                │ stores + per-use bindings
                                ▼
            ┌───────────────────┴────────────────────┐
            │                                        │
   ┌──────────────────┐                ┌────────────────────────────┐
   │   ModelOffloader   │                │        ModelOffloader         │
   │ whole-model DMA  │                │    streamed block mode      │
   └────────┬─────────┘                └─────────────┬──────────────┘
            │                                        │
            │             ┌──────────────────────────┴──────────┐
            │             │  components (ordered):              │
            │             │  • PinnedComponent (non-streamed,   │
            │             │    include names from composition)  │
            │             │  • N × StreamedComponent            │
            │             │                                     │
            │             │  optional LoRA:                     │
            │             │  • post-copy hooks for merge mode   │
            │             └──────────────────────────┬──────────┘
            │                                        │
            └────────────────────┬───────────────────┘
                                 ▼
                      ┌──────────────────┐
                      │ PinnedParam      │  per-parameter pinned-CPU state
                      │ adapter-capable  │  via tensor adapters
                      └──────────────────┘
```

`ResourceStore` is the cache admission contract: it reports
`cache_bytes` for reusable backing storage. `ResourceBinding` is the
per-use lifecycle contract: `value`, `activate(device=None)`, and
`deactivate()`. `ModelStrategy` specializes `ResourceBinding` for
`nn.Module` values and adds `model`.

`ModelCache` caches stores and creates bindings for `use()` calls.
Stores that manage trainable parameters may reject concurrent same-key
bindings. A custom resource fits by passing a store factory and binding
function to `ResourceSpec`:

```python
from torch import nn

class MyStrategy:
    @property
    def model(self) -> nn.Module: ...
    @property
    def value(self) -> nn.Module: ...
    def activate(self, device=None) -> None: ...
    def deactivate(self) -> None: ...

class MyStore:
    @property
    def cache_bytes(self) -> int: ...
    def bind(self) -> MyStrategy: ...

spec = ResourceSpec(
    key="my-resource",
    estimated_cache_bytes=...,
    store_factory=MyStore,
    bind=lambda store: store.bind(),
)
```

A narrower `ModelStrategyComponent` Protocol (just `activate` +
`deactivate`, no `model`) describes pieces composable inside a top-level
strategy — `StreamedComponent` and `PinnedComponent` both satisfy it.

`TensorAdapter` is the per-parameter extension point. Its base contract
only covers inference movement: clone/pin, H2D copy, GPU wrapper rebuild,
cache bytes, logical compute dtype, and block-layout signatures. Extra
behaviors are explicit capabilities: CPU round-trip for optimizer-step
sync, `Parameter.data` swap for trainable streaming, and dense in-place
`addmm_` for activation LoRA merge.

## Store and Binding Lifecycle

Stores own cache accounting. Pinning happens during store construction
so `cache_bytes` is final at admission time; bindings are created for
individual uses:

```
store constructed -> bind -> activate <-> deactivate -> drop binding refs
```

Binding `activate(device=...)` makes the model usable for compute on the
requested device. `ModelOffloader`, `MpsWeights`, `PinnedComponent`, and
`StreamedComponent` require an explicit device. CUDA activation uses the
streaming/DMA path where applicable; CPU activation is pass-through over
pinned host-backed storage.
`deactivate()` releases transient device resources. Store-owned pinned
storage remains cached until the store is evicted or otherwise released.

Construction optimizes peak host memory. Pinning clones managed tensors
into pinned CPU storage. For plain `torch.Tensor` parameters, the source
`Parameter.data` may be immediately repointed at the pinned clone as soon
as that pinned parameter is created. This releases the original source
storage early, avoiding temporarily holding both pageable and pinned
copies for CPU-origin models and promptly freeing GPU storage for
CUDA-origin models. It is a clone-to-pinned plus storage swap, not true
in-place pinning. Tensor subclasses such as quanto, GGUF, and NVFP4 do
not use this `.data` swap when it would lose wrapper state.

**There is no `close()`.** To release pinned host memory, release the
store reference. With `ModelCache`, that means evicting or clearing the
entry. Python's refcount-based GC frees pinned tensors immediately once
the store and any escaped binding/model references that still point at
store-backed tensors are gone.

**Failure semantics.** If construction raises after pinning has started,
the model may already be partially repointed to pinned storage. Treat the
partially constructed store/model as unrecoverable: drop those references
and rebuild from a fresh model instance. If binding `activate()` raises
midway, the binding may contain partial device state; don't retry
`activate()` on that binding. Drop the binding reference and bind again
from the store when the store itself is still valid.
This is a low-level library; we don't guard against caller misuse.

## Compatibility

- **`torch.compile` is not supported** for `ModelOffloader`-managed
  modules. Its `PinnedComponent` swaps parameter registry entries
  (`module._parameters[leaf] = new_param`) on activate/deactivate, and
  `StreamedComponent` registers forward-pre hooks that mutate registered
  parameters on every block call. Both invalidate the tensor-identity assumptions
  `torch.compile` makes about its trace.
- **Wrap before DDP/FSDP**, not after. Those wrappers manage parameter
  storage themselves and conflict with the registry-replacement pattern.
- **Coarse cache concurrency.** `ModelCache` protects cache metadata,
  admission, binding, activation, and deactivation with an internal lock.
  The lock is released while caller code runs inside `cache.use(...)`.
  The cache does not make a yielded model object safe for concurrent
  same-key execution; trainable model specs reject concurrent same-key
  bindings.
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

`ModelOffloader` handles the standard `tie_weights()` pattern (one
`Parameter` referenced under multiple names) plus the rarer case of
distinct quanto wrappers around shared inner `_data` storage.

`ModelOffloader` is intended for ordinary transformer block lists where
the streamed block weights are independent. It does not prevalidate
unusual shared-storage layouts that cross block/non-block boundaries;
use whole-model `ModelOffloader` if that sharing must be preserved.

## Quanto support

Quanto-quantized models (`optimum.quanto.WeightQBytesTensor`) are
handled correctly by both `ModelOffloader` modes. `PinnedParam` decomposes
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

## TorchAO NVFP4 support

TorchAO NVFP4 weights
(`torchao.prototype.mx_formats.nvfp4_tensor.NVFP4Tensor`) are handled as
frozen inference weights when the `nvfp4` optional extra is installed.
`PinnedParam` pins the packed FP4 `qdata`, FP8 block `scale`,
optional per-tensor scales, and the TorchAO dispatch metadata, then
rebuilds the `NVFP4Tensor` wrapper around GPU storage on activation.
The optional extra requires TorchAO plus PyTorch 2.8+; dynamic NVFP4
matmul execution still depends on Blackwell-class CUDA hardware and the
matching PyTorch CUDA stack.
For uv-managed installs on Linux/Windows, this repo routes `torch` and
`torchao` through PyTorch's CUDA 13.0 wheel index. Use
`uv sync --extra nvfp4 --group dev` and then
`pytest tests/test_nvfp4_adapter.py -q -rs` to exercise the optional
TorchAO NVFP4 coverage.

NVFP4 support is intentionally model-weight only. The adapter does not
opt into CPU round-trip, trainable `Parameter.data` swap, or activation
merge-mode LoRA. Use routed LoRA when the target module is a logical
`nn.Linear` with compatible shape and compute dtype. Permanent
`merge_lora()` does not bake into NVFP4 weights.

## Failure modes

The cache and strategies surface failures as typed exceptions rather
than silent corruption.

| Exception | When |
|---|---|
| `ModelTooLargeError` | Cache miss can't fit even after evicting all inactive entries. Exposes `required`, `used`, and `limit`. |
| `EvictionPolicyError` | Custom eviction policy returned duplicate/non-candidate victims or too few bytes |
| `ModelInUseError` | `evict()` / `clear()` / `unregister()` called while entry is active |
| `DuplicateModelKeyError` | `register()` called for an existing key without `replace=True` |
| `ModelNotRegisteredError` | `use(str)` called for an unknown key |

## State Inspection

Use `cache.used_cache_bytes` and `cache.available_cache_bytes` for
current cache accounting. `available_cache_bytes` is signed; negative
means a store reported growth after admission and the cache is currently
over budget. Use `cache.info(key)` for per-key state when needed.
