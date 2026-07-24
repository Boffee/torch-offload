# Memory

A model-agnostic GPU/CPU memory manager for PyTorch. It caches reusable
model and LoRA resources, and swaps independent models in and out of
GPU memory under a policy-driven cache.

Self-contained, library-friendly: no dependencies beyond `torch` (plus
optional `optimum.quanto`, `gguf`, and `torchao` for quantized models). Designed
to be lifted into its own package when a second consumer appears.

## What's in here

| Module | Role |
|---|---|
| `resource_cache.py` | `ResourceCache`, eviction policy, cache metadata, and cache errors |
| `model_cache.py` | `ModelCache` — model-aware `ResourceCache` with activation and LoRA coordination |
| `resource_specs.py` | `ModelSpec`, `LoRASpec`, `ObjectSpec` — standard frozen resource specifications |
| `protocols.py` | `ResourceSpec`, `ResourceStore`, `ResourceBinding` plug-in contracts |
| `block_compile.py` | `BlockCompileConfig` — opt-in Inductor policy for streamed block forwards |
| `model_offloader.py` | `ModelOffloader` — cached single-model runtime for whole-model bulk pinned-CPU↔GPU or streamed block offload |
| `pinned_component.py` | `PinnedComponentStore`, `PinnedComponent` — lower-level reusable pinned backing storage plus lifecycle-only pinned component used by `ModelOffloader` |
| `streamed_component.py` | `StreamedComponentStore`, `StreamedComponent` — lower-level streamed backing storage plus per-block-list streaming component |
| `lora.py` | `LoRA`, `LoRATransform` — cached pinned factors plus merge and routed application hooks |
| `merge.py` | `merge_lora()` — permanent in-place LoRA merge into base weights |
| `pinned_param.py` | `PinnedParam` — per-parameter pinning primitive (handles plain tensors, quanto, GGUF, bitsandbytes, DTensor, and TorchAO dynamic/static scaled-FP8 / INT8 / MX (MXFP8, MXFP4) / NVFP4 / INT4 tile-packed via adapters; see [Quantized weight support](#quantized-weight-support)) |
| `pinned_module.py` | Internal name-keyed pinned module storage plus concrete module bindings |
| `tensor_adapters.py`, `quanto_adapter.py`, `gguf_adapter.py`, `nvfp4_adapter.py`, `mx_adapter.py`, `float8_adapter.py`, `static_float8_adapter.py`, `int8_adapter.py`, `int4_tile_adapter.py`, `dtensor_adapter.py`, `gguf_dequant.py` | Tensor adapter contracts/implementations and optional optimum-quanto / gguf / torchao / DTensor support |
| `torchao_structured_adapter.py` | Internal: shared `TorchaoStructuredAdapter` base for the TorchAO subclass adapters (scaled-FP8 / INT8 / MX / NVFP4 / INT4 tile-packed) — common pin/move/identity mechanics + per-format hooks; capabilities beyond inference movement (CPU round-trip, dequant/requant LoRA merge) are opted into per subclass |
| `dtensor_adapter.py` | Internal: movement-only `DTensorAdapter` for tensor-parallel `DTensor` weights — composes with every other adapter by delegating the local shard to the registry and replaying the `(mesh, placements)` wrapper; frozen-inference scope (see `_dtensor.py`) |
| `tensor_adapter_registry.py` | Public external-adapter registration plus adapter dispatch and tensor-identity helpers |
| `module_names.py` | Internal name traversal and mutation helpers |
| `_quanto.py` | Internal: optimum-quanto optional-import + layout validation; consumed by `quanto_adapter.py` and `merge.py` |
| `_torchao_nvfp4.py` | Internal: TorchAO NVFP4 optional-import + layout validation and dequant/requant; consumed by `nvfp4_adapter.py` |
| `_torchao_mx.py` | Internal: TorchAO MX (MXFP8 / MXFP4) optional-import + layout validation, supported-dtype gate, and dequant/requant; consumed by `mx_adapter.py` |
| `_torchao_float8.py`, `_torchao_static_float8.py` | Internal: TorchAO dynamic/weight-only `Float8Tensor` and calibrated static `PrototypeFloat8Tensor` optional imports, layout validation, and dequant/requant; consumed by the corresponding FP8 adapters |
| `_torchao_int8.py` | Internal: TorchAO INT8 optional-import + layout validation and dequant/requant; consumed by `int8_adapter.py` |
| `_torchao_int4_tile.py` | Internal: TorchAO INT4 tile-packed (CUDA-native tinygemm) optional-import + layout validation; consumed by `int4_tile_adapter.py` |
| `_dtensor.py` | Internal: PyTorch `DTensor` optional-import + mesh/placements signatures and local-shard rewrap; consumed by `dtensor_adapter.py` |

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

1. **Cached resources** that pin reusable model or LoRA state to host RAM.
2. **Activation lifecycles** that move one cached model onto a compute device.
3. **A resource cache** that evicts least-recently-used unleased entries and
   protects leased stores from eviction.
4. **A model-aware cache** that leases model and LoRA resources and owns their
   device lifecycle.

## When to use what

| Situation | Use |
|---|---|
| Most application code, especially multiple models or repeated calls | Use **`ModelCache`** with **`ModelSpec`** |
| Model too big for a CUDA GPU even when active | Use **`ModelSpec(..., blocks_attr=...)`** and pass a **`StreamConfig`** to `ModelCache.use()` |
| LoRA adapters reused across calls | Pass **`LoRASpec`** entries through **`ModelCache.use()`** |
| Low-level/manual lifecycle for one model | Use **`ModelOffloader.from_module(model)`** directly |
| Component or resource development | Use the lower-level store/binding protocols and component stores directly |

## Quick start: cached model use

```python
import torch
from torch_offload import ModelCache, ModelSpec

cache = ModelCache(max_cache_bytes=24 * 1024**3)
model_spec = ModelSpec(
    key="main",
    estimated_cache_bytes=12 * 1024**3,
    factory=build_my_model,  # returns a fresh nn.Module
)
device = torch.device("cuda")

# First use builds and leases the runtime.
with cache.use(model_spec, device=device) as gpu_model:
    output = gpu_model(input_tensor)

with cache.use(model_spec, device=device) as gpu_model:
    output = gpu_model(input_tensor_2)
```

`ModelCache` inherits the complete `ResourceCache` API, adding model activation
and LoRA coordination to the same registry and memory budget. `ModelSpec`
factories should build fresh modules. One model cache entry contains one
`ModelOffloader` and one model instance. Uses are sequential:
an overlapping activation raises `ModelRuntimeInUseError`. Applications that
need concurrent replicas must register separately constructed models under
distinct cache keys, which intentionally duplicates their pinned host storage.
To release pinned host memory, evict or clear inactive cache entries and drop
any escaped model references.

## Manual offloader lifecycle

Use `ModelOffloader` directly when you want explicit lifecycle control without
`ModelCache`.

```python
import torch
from torch_offload import ModelOffloader

model = build_my_model()
offload = ModelOffloader.from_module(model)
device = torch.device("cuda")

offload.activate(device)
try:
    output = offload.value(input_tensor)
finally:
    offload.deactivate()

del offload, model  # drop refs to free pinned host memory
```

`ModelOffloader.from_module()` mutates the source model while
pinning: frozen `nn.Parameter` registry entries get repointed at
Parameters wrapping pinned CPU storage, trainable Parameter objects keep
their identity and point their `.data` at pinned CPU storage, and buffers
are replaced with pinned copies. After construction, only access the bound
model while the offloader is active: call `activate(device)`, use
`offload.value`, and guarantee a matching `deactivate()` with `try`/`finally`.
For CUDA training, wrap `optimizer.step()` in
`offload.optimizer_step()` so trainable GPU updates are copied back to
the pinned CPU cache before deactivation.
**Drop the offloader and model references to release pinned host
memory** — there's no `close()`; resource cleanup is reference-drop + GC.

## Manual block streaming

For models too big to fit on GPU even when active. Streams transformer
blocks through a small GPU-resident window using forward-pre hooks
and a CUDA-stream-based async prefetcher.

```python
import torch
from torch_offload import ModelOffloader, StreamConfig

# Construction pins and binds once; cache_bytes is final immediately.
# blocks_attr selects what streams; the residency policy is supplied per
# activation via StreamConfig, not at construction.
offload = ModelOffloader.from_module(
    model,
    blocks_attr=["transformer_blocks"],  # path(s) to the nn.ModuleList
)
device = torch.device("cuda")

offload.activate(
    device,
    stream_config=StreamConfig(
        num_resident_blocks=1,   # blocks kept on GPU; rest stream in
        num_prefetch_blocks=2,   # GPU pool = resident + prefetch
    ),
)
try:
    output = offload.value(input_tensor)
finally:
    offload.deactivate()

del offload, model  # drop refs to free pinned host memory
```

The residency policy lives on `StreamConfig`, supplied per activation
(`offload.activate(device, stream_config=...)` or
`cache.use(..., stream_config=...)`) — it governs GPU residency, a
runtime concern, and is not part of the pinned backing. `num_resident_blocks=1`
(the default) is right for almost all workloads: eviction is LRU, so a
sequential pass through the blocks reloads every block each iteration
regardless of residency — extra resident slots cost GPU memory without
reducing transfer volume. Spend spare VRAM on `num_prefetch_blocks` instead.
Values above the block count clamp to it, so one config works across models
of different depths. Streaming itself is selected by `blocks_attr`; with no
`blocks_attr` (the default) nothing streams — the whole model is one
bulk-pinned component that activation copies to the GPU.

`ModelOffloader` only streams on CUDA. Activating the binding on
`cpu` is a pass-through over the already-installed pinned CPU storage:
no target pool, no streaming hooks, no weight copies.
`lora_mode="merge"` is CUDA-only; use routed LoRA mode for
CPU activation. Routed LoRA installs target-Linear hooks: a forward-PRE
hook copies that target's pinned factors to the input device, and a
forward-POST hook applies the residual and releases those device copies.

### Optional streamed-block compilation

Repeated streamed blocks can opt into forward-only Inductor compilation at
runtime construction:

```python
from torch_offload import BlockCompileConfig, ModelOffloader

offload = ModelOffloader.from_module(
    model,
    blocks_attr=["transformer_blocks"],
    block_compile=BlockCompileConfig(
        dynamic=True,
        fullgraph=False,
    ),
)
```

`block_compile=None` (the default) preserves eager behavior. One configuration
applies to every `blocks_attr` group, and supplying a compile configuration
without `blocks_attr` raises. The initial API fixes the backend to Inductor's
default mode; CUDA Graph modes and backend-specific options are intentionally
not exposed.

Only each distinct block module's `forward` is compiled. Its module
`__call__` and torch-offload's forward-pre hook stay eager, so block
activation and prefetch finish before compiled computation begins. Compiled
forwards are installed only for CUDA activations and the exact original
forwards are restored on deactivate or activation rollback. CPU activation
remains eager. The lazy compiled callables are retained by the bound runtime,
so later eligible activations can reuse their compiled graphs.

Compilation is inference-only in this initial implementation. Streamed
training remains available without `block_compile`, but combining block
compilation with autograd/checkpointed training is unsupported until it has
dedicated correctness coverage.

Merge-mode LoRA remains compatible. If any routed LoRA is active, every
streamed block runs eagerly for that activation because routed child-Linear
hooks stage parameters inside the block forward. The bypass is temporary:
a later activation with no routed LoRA uses compiled forwards again.
Selecting `lora_mode="routed"` without supplying a LoRA does not bypass
compilation.

Compiler and backend failures propagate normally. torch-offload never catches
a failed compiled invocation and retries that same block eagerly: with graph
breaks, earlier segments may already have executed, so retrying could duplicate
mutations. Dynamo's native graph-break and recompilation-limit behavior still
applies.

By default, trainable parameters (e.g. LoRA adapters) are managed by
the composed `PinnedComponent`: they move to GPU on CUDA activation and
back to pinned CPU storage on deactivate. On CPU activation they stay in
the host-backed module state. Wrap CUDA optimizer updates in
`offload.optimizer_step()` so updated trainable bytes are copied back
to pinned CPU storage before deactivation.

To reduce trainable-weight residency during training, opt into
streaming in-block trainable weights:

```python
offload = ModelOffloader.from_module(
    model,
    blocks_attr=["transformer_blocks"],
    stream_trainable_weights=True,
)
```

During CUDA activation in this mode, only the trainable parameter
`.data` streams. It is GPU-resident while its block is resident, plus
during the optimizer update. CPU activation remains pass-through.
Gradients are not streamed; PyTorch owns `param.grad` normally.

### LoRA merge

`ModelOffloader` supports optional per-weight LoRA merging through activation
arguments. Merge mode
installs activation-scoped post-copy hooks for managed parameter
targets. Unknown targets raise during activation. LoRA target keys must
match the model's parameter names exactly; any remapping — stripping a
`diffusion_model.` prefix, inserting a PEFT `.base_layer.` segment — is
the caller's job when building the LoRA state dict. Each hook runs
immediately after the owning
component copies the base weight from pinned CPU storage to GPU, so both
block-streamed and non-block weights use the same merge path. Merge
compatibility is adapter-owned: plain dense tensors opt into in-place
`addmm_`; structured quantized wrappers can opt into
dequantize/requantize plus `copy_into` updates. Use routed mode for
formats that do not expose either merge capability but still provide a
compatible logical Linear weight shape and compute dtype. `PinnedParam`
remains a storage primitive; LoRA merge mode asks the selected adapter
for the required update capability.

The LoRA request is scoped to one `activate()` call. Target lookup
is resolved during activation; target
compatibility can be preflighted with `LoRATransform.validate_target()`
or validated when the merge hook applies.

```python
import torch
from torch_offload import ModelOffloader, LoRA
from safetensors.torch import load_file

offload = ModelOffloader.from_module(
    model,
    blocks_attr=["transformer_blocks"],
    # Default: stream_trainable_weights=False
)
device = torch.device("cuda")

# Each LoRA owns immutable pinned factors shared by merge and routed uses.
lora_a = LoRA.from_state_dict(
    state_dict=load_file("lora_a.safetensors"),
)
lora_b = LoRA.from_state_dict(
    state_dict=load_file("lora_b.safetensors"),
)

offload.activate(
    device,
    loras=[lora_a, lora_b],
    lora_strengths=[0.8, 0.5],
    lora_mode="merge",
)
try:
    output = offload.value(input_tensor)
finally:
    offload.deactivate()
```

Block reload from pristine pinned CPU storage automatically clears
the previous merge — no explicit unmerge step needed.

Pass `lora_mode="routed"` as an alternative to the default merge mode.
Routed mode installs a forward hook pair on each matched
`nn.Linear` parent — `y = base(x) + alpha * B * A * x` — instead of merging
into the base weight. Its PRE hook copies only that target's factors from
pinned CPU storage to the invocation's input device; its POST hook applies
the residual and releases those device tensors. Multiple LoRAs on one target
are grouped into one hook pair and summed independently. **Routed mode is
inference-only:** factors are frozen (`requires_grad=False`) and no gradient
flows to them. LoRA backing is immutable, so merge and routed uses may overlap
across model runtimes. Use routed mode when:

- The base weight is quantized or otherwise structured, but still exposes
  a logical `nn.Linear` weight shape and compute dtype, and its adapter
  does not support merge updates. `lora_mode="routed"` works because it
  doesn't touch the base.
- You want to switch LoRAs frequently without re-streaming the underlying
  base weight or retaining the whole adapter on GPU.

Routed mode is restricted to `nn.Linear` parents. It handles tied
weights by hooking only the exact parent module named by the target, so
it never mutates shared storage. Packed formats whose parameter shape
differs from the logical matmul weight need a per-format route layer.

For a one-shot **permanent** merge — bake the LoRA into the model
weights and discard the LoRA — use `merge_lora`:

```python
from torch_offload import merge_lora, LoRA

merge_lora(
    model,
    [(LoRA.from_state_dict(state_dict=load_file("lora.safetensors")), 0.8)],
)
```

This uses an in-place `addmm_` for plain fp/bf bases and
dequantize/requantize for quantized bases that expose it (quanto,
bitsandbytes, and TorchAO scaled-FP8 / INT8 / MX / NVFP4 — lossy but
standard); formats without a merge path (GGUF, TorchAO INT4 tile-packed)
need routed LoRA instead. See [Quantized weight
support](#quantized-weight-support) for the full matrix. Unlike an
activation-scoped LoRA request, this is not reversible. Unknown targets
raise, and all target names, factor shapes, and advertised merge
capabilities are preflighted before mutation. Multiple LoRAs for one
quantized parameter are accumulated in dense space and requantized once.

### Heterogeneous block lists

`blocks_attr` accepts a list of dotted paths for models with
multiple kinds of blocks (e.g. Flux's `transformer_blocks` +
`single_transformer_blocks`). Each path becomes its own streaming
group with its own target pool; the streaming settings are shared by
all groups. Blocks within a group must share the same parameter
layout (names/shapes/dtypes/quant-metadata) — split heterogeneous
block lists into separate `blocks_attr` entries. For per-group
streaming settings, compose `StreamedComponentStore` instances
directly:

```python
offload = ModelOffloader.from_module(
    model,
    blocks_attr=["transformer_blocks", "single_transformer_blocks"],
)
# One StreamConfig at activation is shared by all groups, e.g.:
#   offload.activate(device, stream_config=StreamConfig(num_resident_blocks=1))
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
from torch_offload import ModelOffloader

offload = ModelOffloader.from_module(
    model,
    blocks_attr=["transformer_blocks"],
)
device = torch.device("cuda")

model.gradient_checkpointing_enable()  # required for training
model.train()

offload.activate(device)
try:
    gpu_model = offload.value
    for batch in loader:
        loss = gpu_model(**batch).loss
        loss.backward()
        with offload.optimizer_step():
            optimizer.step()
        optimizer.zero_grad()
finally:
    offload.deactivate()
```

Checkpointing every streamed training block is the caller's
responsibility — `ModelOffloader` does not auto-detect or warn about its
absence. It matters most with `stream_trainable_weights=True`, where the
`.data` swap bypasses autograd's version-counter check, so missing
checkpointing can silently corrupt gradients. Verify every streamed
training block is checkpointed (HF `gradient_checkpointing_enable()` or
manual `torch.utils.checkpoint.checkpoint` wrapping).

Wrap CUDA optimizer updates so managed trainable weights are synced back
to pinned CPU storage. With `stream_trainable_weights=True`, this also
materializes streamed trainable weights on GPU while a normal PyTorch
optimizer mutates them:

```python
offload.activate(device)
try:
    gpu_model = offload.value
    for batch in loader:
        loss = gpu_model(**batch).loss
        loss.backward()

        with offload.optimizer_step():
            optimizer.step()

        optimizer.zero_grad()
finally:
    offload.deactivate()
```

This boundary is not optimizer-specific. It runs whatever
`optimizer.step()` does, copies updated trainable data back to pinned
CPU storage, and leaves gradients on GPU.

## Cached model details

`ResourceCache` owns only reusable-resource admission, accounting, leases, and
eviction. `ModelCache` inherits that API and adds dependency leasing, LoRA
attachment, and device activation for model uses.

```python
from torch_offload import (
    LoRASpec,
    ModelCache,
    ModelSpec,
    StreamConfig,
)
from safetensors.torch import load_file

cache = ModelCache(max_cache_bytes=80 * 1024**3)
device = "cuda:0"

text_encoder = ModelSpec(
    key="text_encoder",
    estimated_cache_bytes=12 * 1024**3,
    factory=build_text_encoder,
)
diffusion_model = ModelSpec(
    key="diffusion_model",
    estimated_cache_bytes=24 * 1024**3,
    factory=build_diffusion_model,
    blocks_attr=("transformer_blocks",),
)
style_lora = LoRASpec(
    key="style-lora",
    estimated_cache_bytes=512 * 1024**2,
    factory=lambda: load_file("style.safetensors"),
    dtype=torch.bfloat16,
)

with cache.use(text_encoder, device=device) as enc:
    embeddings = enc.encode(prompt)

with cache.use(
    diffusion_model,
    device=device,
    lora_specs=[style_lora],
    lora_strengths=[0.8],
    lora_mode="routed",
    stream_config=StreamConfig(num_resident_blocks=1),
) as model:
    latent = model(...)
```

The model cache leases LoRA resources before admitting the model resource. An
adapter selected for a use therefore cannot be evicted by that same model admission.
All leases unwind in reverse order if construction or activation
fails. `lora_strengths` defaults to `1.0` per LoRA; when supplied, it must
have the same length as `lora_specs`. Merge and routed uses may share one
cached LoRA across model runtimes because each runtime owns its own hooks and
temporary device copies.

For direct resource access, use a cache lease:

```python
with cache.lease(style_lora) as lora:  # auto-registers on first lease
    targets = lora.targets
```

> **Anti-pattern:** the factory should build a fresh model each call,
> not capture an externally-held one. With `factory=lambda:
> my_kept_model` the cache is no longer the sole owner of the model.
> Always have the factory build the model itself.

`ResourceCache` accepts custom `EvictionPolicy` implementations. The
default is `LRUEvictionPolicy` for unleased host-cache eviction. The
cache builds the eviction candidate set and byte context, then asks the
eviction policy to choose victims; `ResourceCache` still owns validation,
accounting, admission, and release. Policies are called under
the cache lock. `choose_victims()` must return unique keys from
`context.candidates` and enough bytes to satisfy
`context.bytes_to_free`; otherwise `ResourceCache` raises
`EvictionPolicyError` without evicting anything.

## Architecture

```
registration / cache admission
------------------------------
            ResourceSpec protocol
                    |
         ModelSpec / LoRASpec / ObjectSpec
        |
        v
  +-------------+
  | ModelCache  |  extends ResourceCache with model-aware use
  +-------------+
        |
        +-- builds/admit --> ModelOffloader (one model, one runtime)
        |                    |
        |                    +-- PinnedComponent
        |                    |       |
        |                    |       +-- PinnedParam(s)
        |                    |
        |                    +-- StreamedComponent(s)
        |                            |
        |                            +-- PinnedParam(s)
        |
        +-- builds/admit --> LoRA (pinned factors)
        |
        +-- builds/admit --> custom ResourceStore
        |
        +-- chooses inactive victims via EvictionPolicy

ModelCache.use(...)
-------------------
ModelCache
   |
   +-- lease LoRASpec(s), then ModelSpec
   +-- ModelOffloader.activate(loras=...) claims the model runtime
   +-- yield ModelOffloader.value
   +-- ModelOffloader.deactivate() removes LoRA hooks + releases model
```

`ResourceSpec` is the structural registration contract: `key`,
`estimated_cache_bytes`, `build_store()`, and `value(store)`. The standard
specs are independent frozen dataclasses; custom specs can implement the
protocol without inheriting from them. `ResourceStore` is the backing-state
contract and reports `cache_bytes`. A cache lease protects that store from
eviction but does not create or activate a runtime.
`ResourceBinding` is the active-resource lifecycle contract: `value`,
`activate(device=None, **kwargs)`, and `deactivate()`.
`ModelOffloader` is both a cached `ResourceStore` and a `ResourceBinding`;
`LoRA` is an immutable cached `ResourceStore`. It exposes neither an active
lifecycle nor a model-like `value`; merge and routed hooks read its pinned
factor backing directly.

A custom cached resource needs only one spec and one store:

```python
from dataclasses import dataclass

from torch_offload import ResourceSpec, ResourceStore


class MyStore:
    @property
    def cache_bytes(self) -> int: ...


@dataclass(frozen=True)
class MySpec:
    key: str
    estimated_cache_bytes: int

    def build_store(self) -> MyStore:
        return MyStore()

    def value(self, store: ResourceStore) -> MyStore:
        assert isinstance(store, MyStore)
        return store


spec: ResourceSpec[MyStore] = MySpec(
    key="my-resource", estimated_cache_bytes=...,
)

with cache.lease(spec) as store:
    ...
```

`StreamedComponent` and `PinnedComponent` are composable
`activate`/`deactivate` lifecycle pieces (no `value` or `model`) that live
inside a top-level model runtime rather than acting as one themselves.

`TensorAdapter` is the per-parameter extension point. Its base contract
only covers inference movement: clone/pin, H2D copy, GPU wrapper rebuild,
cache bytes, logical compute dtype, and block-layout signatures. Extra
behaviors are explicit capabilities: CPU round-trip for optimizer-step
sync, `Parameter.data` swap for trainable streaming, and — for LoRA merge
— either dense in-place `addmm_` (plain bases) or dequantize/requantize
plus `copy_into` (quantized wrappers).

Downstream tensor subclasses can provide their adapter without adding a
format-specific dependency to torch-offload:

```python
from torch_offload import TensorAdapter, register_adapter


class MyTensorAdapter:
    # Implement the stateless TensorAdapter protocol.
    ...


remove_adapter = register_adapter(MyTensorAdapter)
```

Register adapters during application startup, before constructing models or
pinned resources. `DTensorAdapter` remains the outermost wrapper; registered
adapters are then checked newest-first before the remaining built-ins. This
lets a downstream adapter override a built-in `isinstance` match for a more
specific subclass, and also lets DTensor delegate a custom local shard through
the same registry. `register_adapter()` returns an idempotent removal callable
for tests and scoped integrations.

## Cached resource lifecycle

Cached resources own cache accounting. Pinning happens during construction so
`cache_bytes` is final at admission time; leases protect resources while they
are used. `ModelOffloader` owns one exclusive activation lifecycle. `LoRA`
remains immutable pinned backing throughout its lease:

```
ModelOffloader: construct -> lease -> activate <-> deactivate -> release lease
LoRA:            construct -> lease -> read pinned factors -> release lease
```

`ModelOffloader.activate(device=...)` makes the model usable for compute on the
requested device. Merge hooks copy factors when their base weight is loaded;
routed PRE hooks copy factors for one Linear invocation and routed POST hooks
release them after enqueueing the residual.
`ModelOffloader`, `MpsWeights`, `PinnedComponent`, and
`StreamedComponent` require an explicit device. CUDA activation uses the
streaming/DMA path where applicable; CPU activation is pass-through over
pinned host-backed storage.
`deactivate()` releases transient device resources. Pinned storage remains
cached until its resource is evicted or otherwise released.

Construction optimizes peak host memory. Pinning clones managed tensors
into pinned CPU storage. For plain `torch.Tensor` parameters, the source
`Parameter.data` may be immediately repointed at the pinned clone as soon
as that pinned parameter is created. This releases the original source
storage early, avoiding temporarily holding both pageable and pinned
copies for CPU-origin models and promptly freeing GPU storage for
CUDA-origin models. It is a clone-to-pinned plus storage swap, not true
in-place pinning. Tensor subclasses such as quanto, GGUF, and NVFP4 do
not use this `.data` swap when it would lose wrapper state.

**There is no `close()`.** To release pinned host memory, first let all
leases end, then evict or clear the cache entry. Python's refcount-based
GC frees pinned tensors once the cache and any escaped resource, binding, or
model references are gone.

**Failure semantics.** If construction raises after pinning has started,
the model may already be partially repointed to pinned storage. Treat the
partially constructed resource/model as unrecoverable: drop those references
and rebuild from a fresh model instance. If `activate()` raises, the offloader
rolls back its active components and releases its activation claim, so a later
well-formed activation may retry the same cached resource. Routed LoRA
hook-registration failures remove any hooks already installed; permanent merge
validates all targets before mutation.
This is a low-level library; we don't guard against caller misuse.

## Compatibility

- **`torch.compile` support is deliberately narrow.** Use
  `block_compile=BlockCompileConfig(...)` to compile only declared streamed
  block forwards during CUDA inference. External whole-model
  `torch.compile(model)`, `model.compile()`, compilation of the pinned
  remainder, and compiled streamed training remain unsupported. Routed LoRA
  temporarily bypasses compiled blocks. Compiler code/artifact caches and
  compiler-owned workspace are outside `ResourceCache.cache_bytes`; model
  eviction does not call process-global `torch.compiler.reset()`.
- **Wrap before DDP/FSDP**, not after. Those wrappers manage parameter
  storage themselves and conflict with the registry-replacement pattern.
- **One runtime per cached model.** `ResourceCache` serializes resource
  construction and lease accounting, then releases its lock while a lease is
  held. Each cached `ModelOffloader` owns one model and rejects overlapping
  activation, including calls from different runners. Concurrent replicas
  require distinct `ModelSpec` keys and therefore distinct pinned storage.
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

## Quantized weight support

Every supported weight type can be offloaded (pinned host ↔ GPU
movement). LoRA differs by type: a base can be **merged** into when its
adapter exposes an in-place update path, and **routed** LoRA
(`lora_mode="routed"` — a forward hook) is available for any of them
whose owning module is a logical `nn.Linear` with compatible shape and
dtype, no merge capability required.

| Weight type | Offload | LoRA merge (`mode="merge"` / `merge_lora`) |
|---|---|---|
| Plain bf16 / fp16 / fp32 | ✓ | dense in-place `addmm_` |
| optimum-quanto (int8 / int4 / …) | ✓ | dequant / requant |
| bitsandbytes NF4 / FP4 | ✓ | dequant / requant |
| bitsandbytes int8 | ✓ | dequant / requant |
| TorchAO scaled-FP8 | ✓ | dequant / requant |
| TorchAO static-activation scaled-FP8 | ✓ | fused Triton merge on CUDA; dequant / requant fallback |
| TorchAO INT8 | ✓ | dequant / requant |
| TorchAO MX (MXFP8 / MXFP4) | ✓ | dequant / requant † |
| TorchAO NVFP4 | ✓ | dequant / requant † |
| GGUF (k-quants) | ✓ | — routed only |
| TorchAO INT4 tile-packed | ✓ | — routed only |
| DTensor (tensor-parallel shard) | ✓ | — routed only ‡ |

Notes:

- **Merging into a quantized base is lossy** (dequantize → apply delta →
  re-encode) but standard; choosing merge vs routed is the caller's
  accuracy/latency tradeoff, and is coarser the fewer bits the format has
  (e.g. MXFP4 / NVFP4 at 4 bits). Re-encoding recomputes scales from the
  merged values, and zero-amax blocks are floored to avoid NaN scales.
- **†** MX and NVFP4 store weights in a block-structured packed layout, so
  the standard re-encode (which produces the contiguous layout) cannot fill
  a transposed (non-contiguous `qdata`) target's storage; those raise a
  clear error pointing to routed LoRA. int8 cannot be transposed, and
  scaled-FP8 is unpacked (1 byte/element), so neither is affected.
- **‡** `DTensorAdapter` delegates the local shard to the inner adapter
  only for *movement*; it exposes no merge capability of its own, so a
  merge into a DTensor weight raises (frozen-inference scope). Routed LoRA
  is the intended path for tensor-parallel weights.
- **CPU round-trip** (D2H, for context-free CPU optimizer steps) and
  **trainable `Parameter.data` swap** are separate capabilities: plain
  tensors have both; quanto and both scaled-FP8 representations add CPU
  round-trip; the other quantized formats are movement + (where shown)
  merge only. See the per-format sections below.

## Quanto support

Quanto-quantized models (`optimum.quanto.WeightQBytesTensor`) are
handled correctly by both `ModelOffloader` modes. `PinnedParam` decomposes
the wrapper into its inner `_data` (int8/fp8) and `_scale` (fp16/fp32)
tensors, pins each, and reconstructs the quanto wrapper around the GPU
storage on activation.

A naive `param.data.clone()` on a quanto tensor silently
*dequantizes* it via the dispatch fallback — the explicit decomposition
is required for correctness.

LoRA on quanto bases supports both activation-scoped merge and permanent
`merge_lora()` through dequantize -> addmm -> requantize; neither path
attempts the ineffective native in-place `addmm_` on a
`WeightQBytesTensor`. Use `lora_mode="routed"` when the base must remain
untouched or adapters need to switch without reloading it.

## TorchAO NVFP4 support

TorchAO NVFP4 weights
(`torchao.prototype.mx_formats.nvfp4_tensor.NVFP4Tensor`) are handled
when the `nvfp4` optional extra is installed.
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

NVFP4 weights support merged LoRA: the adapter exposes
dequantize/requantize plus `copy_into`, so both activation-scoped merge mode
and permanent `merge_lora()` re-derive the FP8 (E4M3) block scales — and,
for two-level scaling, the global `per_tensor_scale` — from the merged
values via `NVFP4Tensor.to_nvfp4`. Re-encoding uses the torch reference
path (`use_triton_kernel=False`), which produces the identical
swizzled-scale layout without NVFP4's optional Triton/`mslk` dependency;
the merged bytes are copied into the existing wrapper, which keeps its own
`use_triton_kernel` flag for the forward matmul. Like any merge into a
quantized base it is lossy, and NVFP4's 4-bit grid makes it coarse, so
choosing merge vs routed LoRA is the caller's tradeoff.

The adapter does not opt into CPU round-trip or trainable
`Parameter.data` swap: the quant state lives in the wrapper object, not
its bytes, so NVFP4 weights stay frozen for streaming/training. Routed
LoRA remains the non-destructive alternative when the target module is a
logical `nn.Linear` with compatible shape and compute dtype.

## TorchAO MX (MXFP8 / MXFP4) support

TorchAO MX (OCP microscaling) weights
(`torchao.prototype.mx_formats.mx_tensor.MXTensor`, created by
`quantize_(...)` with an MX inference config or directly via
`MXTensor.to_mx`) are handled when the `torchao` optional extra is
installed. A single adapter covers both
MXFP8 (`float8_e4m3fn` / `float8_e5m2`) and MXFP4
(`float4_e2m1fn_x2`), since TorchAO models them as the same `MXTensor`
subclass parameterized by `elem_dtype`. `PinnedParam` pins the packed
`qdata`, the E8M0 block `scale`, and the TorchAO dispatch metadata
(`elem_dtype`, `block_size`, `kernel_preference`, `act_quant_kwargs`,
`is_swizzled_scales`), then rebuilds the `MXTensor` wrapper around GPU
storage on activation. MXFP6 and any other MX element dtype are not
admitted; such a tensor falls through to a clear "no adapter" error
rather than being silently mishandled. MX matmul execution still
depends on Blackwell-class CUDA hardware and the matching PyTorch CUDA
stack. Use `uv sync --extra torchao --group dev` and then
`pytest tests/test_mx_adapter.py -q -rs` to exercise the coverage.

MX weights support merged LoRA: the adapter exposes dequantize/requantize
plus `copy_into`, so both activation-scoped merge mode and permanent
`merge_lora()` work by dequantizing to dense, applying the delta, and
re-encoding the power-of-two (E8M0) block scales through the public
`MXTensor.to_mx`. Both element dtypes are mergeable, but MXFP4's 4-bit
grid makes a requantized merge far coarser than MXFP8 — choosing merge vs
routed LoRA is the caller's accuracy/latency tradeoff. The adapter does
not opt into CPU round-trip or trainable `Parameter.data` swap: like
NVFP4 the wrapper's quant state lives in the object, not its bytes, so MX
weights stay frozen for streaming/training. Routed LoRA remains available
when the target module is a logical `nn.Linear` with compatible shape and
compute dtype.

## TorchAO scaled FP8 support

TorchAO scaled-fp8 weights (`torchao.quantization.Float8Tensor`, created
by `quantize_(..., Float8WeightOnlyConfig/Float8DynamicActivationFloat8WeightConfig)`)
are handled when the `fp8` optional extra is installed. `PinnedParam`
pins the fp8 `qdata` and fp32 `scale` tensors plus the TorchAO dispatch
metadata (`block_size`, `mm_config`, `kernel_preference`,
`act_quant_kwargs`), then rebuilds the `Float8Tensor` wrapper around GPU
storage on activation. Per-row and per-tensor scale granularities are
supported; fp8 matmul execution requires SM89+ (Ada/Hopper or newer)
CUDA hardware.

Like MX, INT8, and NVFP4, scaled-fp8 weights support merged LoRA: the
adapter exposes dequantize/requantize plus `copy_into`, so both
activation-scoped merge mode and permanent `merge_lora()` work by
dequantizing to dense, applying the delta, and re-encoding with
recomputed scales through the public `Float8Tensor.from_hp` (lossy but
standard practice for merges into quantized bases). The GPU
representation is byte-identical to the host one, so the CPU round-trip
capability is also available. Trainable `Parameter.data` swap is not —
scaled-fp8 weights stay frozen.

TorchAO's calibrated static-activation representation is handled separately
by `StaticFloat8Adapter`. It targets only
`torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor.PrototypeFloat8Tensor`
weights with per-tensor weight and activation quantization, and pins the FP8
`qdata`, weight `scale`, and checkpoint-provided `act_quant_scale`. All three
are included in identity, block-pool layout compatibility, cache accounting,
H2D/D2H movement, and wrapper reconstruction; the ordinary `Float8Tensor`
adapter remains the weight-only/dynamic path.

TorchAO 0.17 normally requires the activation scale rank to equal the input
rank. torch-offload's static adapter installs a narrow `nn.Linear` dispatch
shim that flattens ordinary activations before static quantization and reshapes
the result afterwards. A checkpoint scalar (or any one-element scale layout)
therefore works unchanged for both 2-D and 3-D Linear inputs. LoRA merge uses
a format-specific Triton kernel pipeline on CUDA when Triton is available,
independently of `block_compile`. It fuses dequantization, the low-rank GEMM,
addition, and tile-level maximum collection before reducing the new per-tensor
weight scale and requantizing. CPU merges and CUDA installations without
Triton retain the generic adapter path. Both paths copy only the re-encoded
weight bytes and scale into the target; the calibrated activation scale is
preserved exactly. Routed LoRA is supported as the non-destructive alternative.
Output activation quantization and non-per-tensor Prototype layouts are outside
this adapter's contract and are rejected explicitly.

## Failure modes

The cache and bindings surface failures as typed exceptions rather
than silent corruption.

| Exception | When |
|---|---|
| `ResourceTooLargeError` | Cache miss can't fit even after evicting all inactive entries. Exposes `required`, `used`, and `limit`. |
| `EvictionPolicyError` | Custom eviction policy returned duplicate/non-candidate victims or too few bytes |
| `ResourceLeasedError` | A cache mutation targets a currently leased entry |
| `ResourceCachedError` | `unregister(..., evict=False)` targets an entry with a built store |
| `ModelRuntimeInUseError` | Any caller overlaps activation of the same cached `ModelOffloader` |
| `DuplicateResourceKeyError` | `register()` is called for an existing key without `replace=True` |
| `ResourceNotRegisteredError` | `lease(str)` is called for an unknown key |

## State Inspection

Use `cache.used_cache_bytes` and `cache.available_cache_bytes` for
current cache accounting. Use `cache.info(key)` for per-key state when
needed.
