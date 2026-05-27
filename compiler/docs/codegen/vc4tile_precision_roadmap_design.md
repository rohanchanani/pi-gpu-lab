# VC4Tile Precision Roadmap: 32-Bit-First Ergonomic VC4Tile with Future-Proof Sub-32 Precision Hooks

**Status:** locked planning decision for the VC4Tile ergonomic milestone and the first end-to-end producer-lowering drafts.  
**Scope of this document:** precision, packing, quantization, and mixed-precision support strategy across VC4Tile, SSAVC4, scheduled VC4, runtime/artifacts, and future Triton/IREE lowering.  
**Immediate implementation policy:** M5 is **only ergonomic VC4Tile**. M5 does **not** implement Triton lowering, IREE lowering, or executable sub-32 precision. M5 should be 32-bit-only in executable semantics, while adding forward-looking metadata fields and diagnostics so later precision work can be added cleanly.

---

## 1. Original question and decision target

The original question was not simply “can VC4 support lower precision?” It was:

> VC4 has hardware packing and sub-32 movement/conversion features. Modern Triton, JAX, PyTorch, and IREE pipelines increasingly expose fp8, fp4, int8, int4, mixed precision, quantization, and scale/zero-point semantics. Should this be added before M5, during M5, after M5, or at a different compiler level? Where should the semantics live so the compiler is not forced into brittle decisions later?

The clarified milestone ordering is:

```text
M5: ergonomic VC4Tile only
    - no Triton lowering
    - no IREE lowering
    - no executable sub-32 precision
    - 32-bit-only executable semantics
    - add forward-looking precision metadata/fields/diagnostics

Post-M5: Triton v1
    - end-to-end producer lowering into VC4Tile
    - 32-bit-only
    - reject/diagnose unsupported sub-32 producer types and precision modes

Post-Triton-v1: IREE/JAX/PyTorch v1
    - end-to-end IREE path into VC4Tile
    - 32-bit-only
    - reject/diagnose unsupported quantized/sub-32 forms

After end-to-end producer paths work:
    - incrementally implement sub-32 storage, packing, conversion, quantization, and eventually specialized low-precision compute where VC4 actually supports it
```

The key decision is:

**Start with 32-bit-only execution through M5, Triton v1, and IREE/JAX/PyTorch v1. During M5, add precision-aware syntax/metadata and strict diagnostics, but do not lower non-32-bit precision to hardware. After the first 32-bit end-to-end producer paths work, add sub-32 precision as a dedicated vertical milestone family.**

This decision preserves momentum toward end-to-end compiler functionality while preventing the VC4Tile interface from becoming precision-blind.

---

## 2. High-level decision summary

### 2.1 What is locked now

M5 should add ergonomic VC4Tile features using **32-bit executable semantics only**. The supported executable data carriers are:

```text
i32 / u32 scalar-like values
f32 scalar-like values
vector<16xi32>
vector<16xf32>
32-bit global/shared element accesses
32-bit tile load/store/copy/contract/reduce behavior
```

M5 should still introduce **forward-looking precision markers** on relevant ergonomic surface concepts, especially `tile_load`, `tile_store`, `copy_tile`, `tile_contract`, `tile_dot`, `tile_matmul`, block/tile reductions, shared/VPM copy descriptions, and tile layout descriptors. These markers must default to the current 32-bit behavior.

The fields should exist so later producer lowering can preserve intent such as:

```text
storage_type       = f32 | i32 | u32 | future(f16, u8, i8, fp8, int4, ...)
expressed_type     = f32 | i32 | u32 | future(...)
accumulator_type   = f32 | i32 | u32 | future(...)
precision_policy   = exact_32 | future(storage_dequantize_then_f32, f16_storage_f32_compute, ...)
quantization       = none | future(uniform_affine, symmetric, block_scaled, ...)
scale_granularity  = none | future(per_tensor, per_axis, per_row, per_channel, per_block, ...)
zero_point_policy  = none | future(absent, symmetric_zero, affine, dynamic, ...)
packing            = none | future(vpm_packed, vpm_laned, byte_packed, nibble_packed, ...)
```

For M5, all non-default/non-32-bit values must fail with deterministic diagnostics. They must not be ignored, silently widened, or faked.

### 2.2 What is explicitly not locked yet

This document does **not** lock the final exact spelling of every MLIR attribute/type. The design policy is locked; the exact ODS spelling may still be refined in the M5 package. However, the implementation must preserve these semantic categories:

```text
storage type        how values are represented in memory/VPM/register files before conversion
expressed type      mathematical type after decode/dequantize/unpack
accumulator type    type used by reductions/dot/contract accumulation
precision policy    how to move between storage, expressed, and accumulator domains
quantization data   scale, zero point, granularity, and block/group structure
layout/packing      physical tile/VPM/global organization intent
```

### 2.3 Why this is the right compromise

This gives us the good parts of early precision planning without the risks of early precision implementation.

It avoids these failure modes:

* implementing Triton/IREE before VC4Tile ergonomics exist;
* making VC4Tile surface ops precision-blind and then having to redesign them later;
* prematurely pretending VC4 has modern fp8/int8 tensor-core-like compute;
* burying pack/unpack decisions only in the emitter, too late for copy planning;
* accepting unsupported producer precision modes and silently compiling wrong kernels.

It enables these later steps:

* Triton v1 and IREE v1 can lower 32-bit kernels without waiting for sub-32 support;
* unsupported sub-32 producer IR can produce precise diagnostics instead of falling through;
* when sub-32 support is added, the surface/core boundary already has a place for the semantics;
* copy planning can eventually use `storage_type`, `layout`, and `packing` to choose VPM/VDW/TMU modes;
* `tile_contract` / `tile_dot` / reductions can later consume quantization and accumulator policies in a controlled way.

---

## 3. Evidence: what VC4 actually supports

The VC4 hardware evidence points to a very specific model: VC4 has real sub-32 **storage, movement, pack/unpack, and some packed-byte ALU behavior**, but it does not have modern ML tensor-core-like fp8/int8/fp4/int4 matrix compute.

### 3.1 QPU ALU model: mostly 32-bit carriers

The VideoCore IV QPU has two ALUs, ADD and MUL. The architecture guide describes the ALUs as operating internally on 32-bit integer or floating-point data, with an exception for vectors of four 8-bit quantities. It also says the QPU has hardware to read 16-bit and 8-bit data from register file A, extend or convert that data before feeding the ALUs, and reconvert 32-bit ALU output to 16/8-bit data when writing back to register file A. Accumulators r0-r3 only operate on 32-bit data and have no pack/unpack functionality. This is the core reason we should keep 32-bit carrier values as the default SSAVC4/core representation. [VC4-ARG]

Implication for the compiler:

```text
Do not model ordinary VC4 compute as vector<16xi8> or vector<16xf16> in core.
Use 32-bit carrier values for most arithmetic.
Represent sub-32 as storage/conversion/packing semantics around those carriers.
```

### 3.2 Regfile-A pack/unpack is real and important

The architecture guide’s pack/unpack section says the A-regfile unpack block can convert packed 8-bit or 16-bit data to 32-bit values ready for ALU use. It also says the A-regfile pack block can pack 32-bit ALU results back into 8-bit or 16-bit A-regfile data. Regfile-A unpack includes 16-bit half selection, f16-to-f32 behavior for float-consuming instructions, signed-int16-to-signed-int32 behavior otherwise, and 8-bit byte selection with either normalized color-to-f32 or unsigned-int8-to-int32 behavior. Regfile-A pack can write 16-bit halves or 8-bit bytes, with saturation/format-specific behavior. [VC4-ARG]

Implication for the compiler:

```text
Sub-32 support must be represented before scheduled VC4, because instruction selection and scheduling need to know whether an ALU input/output should use pack/unpack modes.
But the high-level VC4Tile surface should not expose raw pack-bit encodings.
```

### 3.3 R4 unpack and MUL pack are narrower than generic quantization

The guide describes accumulator r4 as having a limited unpack unit for values returned by texture/tile-buffer-like units. R4 unpack supports f16-to-f32 and 8-bit color-to-f32. The MUL ALU can convert a floating-point result to 8-bit color form in the normalized range `[0, 1.0]`, including saturation/rounding behavior. [VC4-ARG]

Implication for the compiler:

```text
Do not conflate VC4 8-bit color conversion with generic signed int8, fp8, or quantized ML storage.
The MUL pack path is useful, but it is not a generic FP8/INT8 quantization engine.
```

### 3.4 VPM/VCD/VDW are the main reason sub-32 matters

The guide describes VPM as a 2D array of 32-bit words, 16 words wide with a maximum height of 64 words from the QPU perspective. The QPU can read/write horizontal or vertical 16-way vectors of 32-, 16-, or 8-bit data. For 16-bit and 8-bit vectors, VC4 supports two split modes: `laned`, where each 32-bit word is split into two 16-bit lanes or four 8-bit lanes, and `packed`, where a sub-vector is taken from successive 32-bit words. [VC4-ARG]

The VPM generic block read/write setup fields encode orientation, size, stride, packed-vs-laned mode, and address/sub-vector selectors. The VDW DMA store setup supports 32-bit, packed 16-bit halfword-offset modes, and packed 8-bit byte-offset modes. VDR DMA load setup likewise has width and byte/halfword selection fields. [VC4-ARG]

Implication for the compiler:

```text
The copy planner is central.
Sub-32 precision is not just an arithmetic feature; it is a memory/VPM/layout feature.
The natural first sub-32 milestone should be storage and copy planning, not tensor-core-style compute.
```

### 3.5 Packed-byte ALU operations exist but are limited

The instruction set includes packed 8-bit operations such as `v8adds`, `v8subs`, `v8min`, `v8max`, and `v8muld`. These operate on four 8-bit quantities packed inside each 32-bit lane. This is useful for some image/byte-vector kernels and potentially some quantized epilogues, but it is not equivalent to a modern int8 dot-product matrix unit. [VC4-ARG]

Implication for the compiler:

```text
Treat packed-byte ALU as a later specialized optimization path.
Do not make it the default lowering target for quantized matmul or tile_contract.
```

### 3.6 Hardware conclusion

VC4 gives us these sub-32 capabilities:

```text
Strong / should support eventually:
  - f16 storage with f32 compute carriers
  - u8/i8-like storage with explicit extension/dequantization
  - 8/16-bit VPM reads/writes
  - VPM packed/laned layouts
  - VDW/VDR packed byte/halfword transfer modes
  - A-regfile pack/unpack
  - selected packed-byte ALU ops

Weak / must not pretend native:
  - native FP8 matrix math
  - native FP4 matrix math
  - native INT4 matrix math
  - tensor-core-like INT8 dot accumulation
  - arbitrary low-precision MMA
```

That means the correct compiler approach is:

```text
storage + movement + explicit convert first
quantize/dequantize next
specialized packed-byte compute later
modern fp8/fp4/int4 as storage/software-emulated paths only until proven otherwise
```

---

## 4. Evidence: what upstream producer systems expose

Modern upstream systems expose precision in ways that are richer than VC4’s raw hardware encodings. They generally describe **storage type, expressed type, accumulator type, scale/zero-point, granularity, and layout**, not raw target pack bits.

### 4.1 Triton

Triton’s `tl.dot` documentation accepts inputs with scalar types including `int8`, `float8_e5m2`, `float16`, `bfloat16`, and `float32`, and exposes `input_precision` and `out_dtype` on the dot operation. This means a future Triton lowering cannot be precision-blind: even if v1 only supports f32/i32, the lowering has to inspect and reject unsupported dot precisions deterministically. [Triton-dot]

Triton’s `tl.dot_scaled` exposes a more modern microscaling model. It accepts lhs/rhs representing fp4, fp8, or bf16 elements; fp4 values can be packed into uint8 inputs; fp8 may be represented as uint8 or fp8 type; separate lhs/rhs scale tensors are passed; and formats include `e2m1`, `e4m3`, `e5m2`, `bf16`, and `fp16`. The docs also state that software emulation may upcast microscaled inputs to bf16/fp16 on hardware without native microscaling support. [Triton-dot-scaled]

The Triton block-scaled matrix multiplication tutorial further demonstrates the direction of travel: fp4/fp8/microscaling support is tied to specific packed layouts, scale-factor layouts, and hardware matrix-core capabilities on newer NVIDIA/AMD GPUs. [Triton-block-scaled]

Implication for our compiler:

```text
Triton v1 should be 32-bit-only.
But the future VC4Tile surface must have fields for storage type, accumulator type, scale, format, and packing/layout.
Unsupported tl.dot/tl.dot_scaled forms must be diagnosed, not silently widened.
```

### 4.2 IREE and MLIR Linalg

IREE’s Linalg tutorial frames Linalg as an MLIR middle-end dialect and describes `linalg.generic` as general enough to express usual ML workloads in any quantization scheme, high-level enough to lower to efficient target code, and a good fit for IR-to-IR transformations. It also describes IREE as a compiler/runtime that lowers MLIR programs through successive lower-level dialects to machine code for CPU, GPU, and other targets. [IREE-Linalg]

IREE’s pass documentation includes passes such as fusing dequantization and matmul linalg.generic ops and generalizing named Linalg ops to generics. This indicates that quantized producer graphs may appear as dequantize/matmul patterns rather than a single target-specific low-precision op. [IREE-passes]

Implication for our compiler:

```text
IREE/JAX/PyTorch v1 should be 32-bit-only.
But later IREE lowering needs to recognize quantization as explicit IR semantics: dequantize, quantize, scale, zero point, matmul/reduction, and layout.
VC4Tile should preserve enough metadata to map those patterns to storage/copy/contract decisions later.
```

### 4.3 StableHLO and MLIR Quant

StableHLO quantization describes uniform quantization as mapping floating-point values to integers through scale and zero point. It supports per-tensor quantization and per-axis quantization, where slices along a quantized dimension have distinct scale/zero-point parameters. [StableHLO-quant]

MLIR’s Quant dialect supports uniform quantized types with stored type, expressed type, scale, zero point, per-channel parameters, and sub-channel/blockwise quantization. The sub-channel form divides a tensor into blocks, each with its own scale and zero point; tensor elements use the scale/zero point selected by their block. [MLIR-Quant]

Implication for our compiler:

```text
The future quantization model must not be just “dtype = i8”.
It must represent storage type, expressed type, scale, zero point, and granularity.
The later VC4 precision milestone must handle per-tensor/per-axis/per-row/per-channel/per-block metadata as semantic inputs, even if only a subset lowers to optimized hardware forms initially.
```

### 4.4 PyTorch / torchao

Torchao’s quantized inference documentation lists dynamic and weight-only quantization workflows for linear layers across float8, int8, mxfp8/mxfp4, int4/intx, nvfp4, and mixed configurations. It distinguishes activation dtype from weight dtype and describes granularities such as per-tensor, per-row, per-token, per-channel, per-group, and per-block/double-quantized scaling. [TorchAO]

Implication for our compiler:

```text
JAX/PyTorch-through-IREE v1 should not try to support all of this.
But the VC4Tile interface must not make later support impossible.
The important future-proof categories are: activation storage type, weight storage type, accumulator type, scale granularity, zero-point policy, and packed layout.
```

### 4.5 JAX

JAX `matmul` exposes `precision` and `preferred_element_type`, where `preferred_element_type` can request the accumulation/result type. JAX’s `default_matmul_precision` setting controls precision for matmul and convolution on 32-bit inputs and can specify accumulation algorithms/precision presets on platforms that support such choices. [JAX-matmul] [JAX-precision]

Implication for our compiler:

```text
Even in 32-bit-only v1, producer lowering should preserve or diagnose precision requests.
For now, supported JAX/IREE precision should be effectively f32/i32 exact/default on VC4.
Unsupported precision hints must not silently choose a different numeric contract.
```

---

## 5. Core compiler design policy

### 5.1 Separate storage, expressed, accumulator, and carrier types

The compiler must distinguish these concepts:

```text
storage_type:
  The physical representation in global memory, VPM, register-file packed data, or packed buffers.
  Examples: f32, i32, u32, future f16, u8, i8, fp8_e4m3, fp8_e5m2, int4, fp4.

expressed_type:
  The mathematical value after unpack/decode/dequantize.
  Examples: f32, i32, future f16/bf16/f32 expressed values for low-bit storage.

accumulator_type:
  The type used for dot/contract/reduction accumulation.
  Examples: f32, i32, future f32 accumulation for f16/u8/fp8 storage, i32 for int8-style products.

carrier_type:
  The actual compiler value type used in VC4Tile core/SSAVC4 for most arithmetic.
  For VC4, usually i32/f32/vector<16xi32>/vector<16xf32>.
```

M5 executable behavior should support only 32-bit storage/expressed/accumulator combinations. But the surface should make these fields visible so later producer lowering can carry intent.

### 5.2 Do not make sub-32 carrier values the core representation

The default core should not become `vector<16xi8>` or `vector<16xf16>`. VC4’s ALU and current SSAVC4 lowering are fundamentally organized around 32-bit carrier values, with pack/unpack and VPM/VDW modes at boundaries. Sub-32 support should be represented as storage/conversion/layout semantics around 32-bit carriers.

Policy:

```text
VC4Tile surface may expose typed tiles/fragments carrying storage/expressed/accumulator metadata.
VC4Tile canonicalization may use those types/attrs for planning.
VC4Tile core and SSAVC4 should keep ordinary compute in 32-bit carriers unless a later specialized packed-byte optimization path explicitly proves otherwise.
```

### 5.3 Do not bury precision only in QASM emission

Packing is too important to defer until final emission. VPM layout, VDW/VDR modes, and register pack/unpack all affect copy planning, instruction selection, scheduling, and hardware verification.

Policy:

```text
VC4Tile surface: semantic precision/layout metadata.
VC4Tile core: explicit storage/element-size/packing/conversion intent.
SSAVC4: machine-level pack/unpack/VPM/VDW/TMU attrs/ops.
Scheduled VC4: exact instruction fields and setup words.
Emitter: final QASM and runtime artifacts from scheduled VC4, not semantic invention.
```

### 5.4 Treat metadata-only fields as strict

Forward-looking fields must be strict even before they lower to hardware.

M5 should allow:

```text
storage_type = f32/i32/u32/none-default-meaning-32-bit
expressed_type = f32/i32/u32/default
accumulator_type = f32/i32/u32/default
precision_policy = exact_32/default
quantization = none/default
scale_granularity = none/default
zero_point_policy = none/default
packing = none/default-32-bit
```

M5 should reject:

```text
storage_type = f16/u8/i8/fp8/int4/fp4/...
precision_policy = dequantize/quantize/block_scaled/...
quantization != none
scale/zero-point operands on executable paths
packing requiring VPM packed/laned sub-32 behavior
```

The diagnostic should say that M5 is 32-bit-only and that sub-32 precision is reserved for the later precision milestone.

### 5.5 No silent widening

Do not silently widen f16/u8/int8/fp8/int4/fp4 producer values to f32 and call that support. Widening is acceptable only when it is explicitly part of a documented precision policy and tested against a CPU reference.

For M5/Triton-v1/IREE-v1:

```text
unsupported sub-32 input -> deterministic compiler error
unsupported quantization -> deterministic compiler error
unsupported precision request -> deterministic compiler error
```

---

## 6. Expected changes by compiler level

### 6.1 Documentation and milestone specs

M5 docs/specs should state:

```text
M5 is ergonomic VC4Tile only.
M5 executable semantics are 32-bit only.
M5 adds forward-looking precision metadata fields.
M5 must reject unsupported sub-32/quantized fields deterministically.
M5 does not lower Triton or IREE.
M5 does not implement executable f16/u8/i8/fp8/int4/fp4 behavior.
```

Verification specs should require:

* roundtrip tests for default 32-bit precision fields;
* canonicalization tests showing defaults survive or are normalized;
* invalid tests for f16/u8/int8/fp8/int4/fp4 fields;
* invalid tests for non-`none` quantization metadata;
* invalid tests for scale/zero-point operands when no precision milestone has implemented them;
* no changes to lower-half QASM/emitter/runtime behavior for non-32 precision.

### 6.2 VC4Tile surface dialect

M5 should add precision-aware fields to ergonomic ops, but only support 32-bit values.

Candidate surface concepts:

```text
vc4tile.tile_load
vc4tile.tile_store
vc4tile.copy_tile
vc4tile.tile_contract / tile_dot / tile_matmul
vc4tile.tile_reduce / block_reduce / row_reduce
vc4tile.tile_view / shared_tile_view / global_tile_view
vc4tile.layout / shape / role descriptors
```

Candidate metadata categories:

```text
storage_type       default f32/i32/u32 according to result/value type
expressed_type     default equal to storage_type for 32-bit v1
accumulator_type   default f32 for f32 compute, i32/u32 for integer compute if applicable
precision_policy   exact_32
quantization       none
scale_granularity  none
zero_point_policy  none
packing            none/default_32bit
```

In M5 these fields are semantic declarations only for the default 32-bit case. They should be validated and normalized but not cause sub-32 codegen.

### 6.3 VC4Tile canonicalization

M5 should introduce or extend the canonicalization pipeline so ergonomic VC4Tile surface ops lower to VC4Tile core ops before `--convert-vc4tile-to-ssavc4`.

Precision handling in M5 canonicalization:

```text
Input:
  surface ops with optional precision metadata

Validation:
  confirm all executable precision choices are 32-bit/default
  reject unsupported non-32 values with precise diagnostics
  confirm quantization metadata is none/empty

Output:
  VC4Tile core ops with normalized 32-bit metadata or no explicit precision attrs
  no sub-32 pack/unpack/conversion ops
  no scale/zero-point semantics
```

Later precision milestones will extend this stage to produce explicit core conversion/copy/packing operations.

### 6.4 VC4Tile core

M5 VC4Tile core remains 32-bit executable. It may add normalized metadata slots on core ops, but those slots must either be default/exact_32 or rejected.

Existing-ish core concepts should remain compatible with future precision:

```text
masked_load_global
masked_store_global
shared_load
shared_store
copy_tile / copy-to-shared / copy-from-shared
reduce / rotate / barrier / identities
cf + block args after core CFG legalization
```

Future precision milestones may add:

```text
elem_bits = 8/16/32
storage_encoding = raw_u8/raw_i8/f16/fp8_e4m3/fp8_e5m2/int4/...
packing_mode = vpm_packed/vpm_laned/byte_packed/nibble_packed
quantize/dequantize core ops or attrs
scale/zero-point operands
copy-planner choices for VPM/VDW/TMU paths
```

But M5 should not require these to execute.

### 6.5 SSAVC4

M5 should not have to change SSAVC4 for sub-32 execution. If M5 adds metadata plumbing, it should not change scheduled output or hardware behavior.

Future precision milestones will likely add or extend SSAVC4 with:

```text
ssavc4.unpack / pack attrs or ops for regfile-A and r4/MUL-pack paths
ssavc4.vpm.read/write elem_bits + packed_or_laned + orientation + address fields
ssavc4.vdw.store 8/16/32-bit mode fields and byte/halfword offsets
ssavc4.vdr.load 8/16/32-bit mode fields and byte/halfword offsets
explicit dequantize/quantize lowering patterns
possibly specialized packed-byte ALU patterns for v8adds/v8subs/v8min/v8max/v8muld
```

Policy:

```text
SSAVC4 should be the first level that knows about VC4 machine pack/unpack modes.
VC4Tile surface should know semantic precision, not raw hardware encoding.
```

### 6.6 Scheduled VC4

Scheduled VC4 will eventually carry exact QPU instruction attributes:

```text
ALU pack/unpack bits
pm bit meaning
regfile A pack/unpack selection
r4 unpack selection
MUL pack selection
VPMVCD setup words
VDW/VDR mode fields
hazard-safe scheduling around pack/unpack/peripheral accesses
```

M5 should not introduce scheduled VC4 sub-32 semantics. Later precision milestones should include scheduled-artifact tests that check exact pack/unpack and VPM/VDW setup behavior.

### 6.7 Artifact emitter and runtime

The emitter should continue emitting from scheduled VC4, not inventing semantic precision decisions.

For M5:

```text
No sub-32 emitter changes are expected.
No runtime ABI changes are expected.
No generated artifact changes are expected except those caused by ergonomic 32-bit VC4Tile lowering.
```

Future precision milestones may need:

```text
manifest metadata for element-size/storage-format if useful for testing/debugging
kernel_launch packing helpers only if scalar/user argument types need sub-32 handling
hardware fixtures with byte/halfword CPU reference comparisons
strict no-synthetic-result scans
```

Device pointers are byte addresses already, so many buffer ABI cases may not require ABI shape changes. But generated launch wrappers and test harnesses will need to know how many bytes to allocate/copy and how to construct CPU references for sub-32 storage.

### 6.8 Producer lowering: Triton and IREE

M5 has no producer lowering. But M5 should be designed so later producer lowering can map precision fields cleanly.

Triton v1:

```text
Support only f32/i32/u32 producer data paths.
Reject tl.dot/tl.dot_scaled/fp8/fp4/int8/int4/f16/bf16 paths unless a 32-bit-only fallback is explicitly part of the v1 spec.
Preserve enough diagnostic context to explain which precision is unsupported.
Map 32-bit tile_load/store/contract/reduce to ergonomic VC4Tile surface ops.
```

IREE/JAX/PyTorch v1:

```text
Support only 32-bit Linalg/dispatch paths.
Reject quantized types, dequantize+matmul patterns, int8/int4/fp8/fp4 storage, and unsupported precision hints.
For JAX precision/preferred_element_type, support only the cases whose semantics match the VC4 32-bit implementation.
```

Later producer precision support:

```text
Map StableHLO/MLIR Quant/PyTorch/IREE quantized semantics into VC4Tile storage/expressed/accumulator/scale/zero-point fields.
Map Triton dot_scaled into VC4Tile scale/format/packing metadata.
Decide whether each form lowers to optimized VC4 packing, explicit software conversion, or a diagnostic.
```

---

## 7. Interplay between load/store/copy and compute primitives

Precision is not just a `tile_dot` issue. It flows through the tile pipeline.

### 7.1 `tile_load`

`tile_load` should represent reading a logical tile/fragment from global/shared memory.

M5 behavior:

```text
32-bit storage only
storage_type == expressed_type == carrier type
no dequantize
no unpack
```

Future behavior:

```text
storage_type may be f16/u8/i8/fp8/int4/etc.
expressed_type may be f32/i32/etc.
quantization metadata may provide scale and zero point.
copy planner may choose TMU, VPM/VDR, or direct load paths.
canonicalization may insert unpack/dequantize to 32-bit carriers.
```

### 7.2 `copy_tile`

`copy_tile` is where VC4 hardware-specific layout choices become powerful.

M5 behavior:

```text
32-bit tile copies only
layout metadata may exist but must describe supported 32-bit layouts
no packed/laned sub-32 VPM modes
```

Future behavior:

```text
choose VPM packed vs laned modes
choose horizontal vs vertical VPM orientation
choose VDW/VDR byte/halfword modes
perform layout conversions between global, shared VPM, and register carriers
handle f16/u8/i8 storage density
```

This is likely the most important sub-32 implementation area because VC4’s VPM/VDW/VDR machinery is rich and tile-shaped.

### 7.3 `tile_store`

`tile_store` should represent writing a logical tile/fragment back to memory.

M5 behavior:

```text
32-bit stores only
no quantize
no pack
```

Future behavior:

```text
quantize f32/i32 carriers to u8/i8/int4/fp8/f16 storage where supported
pack to VPM or regfile A as needed
use VDW packed 8/16-bit modes for global stores
validate saturation/rounding semantics against CPU references
```

### 7.4 `tile_contract`, `tile_dot`, `tile_matmul`

These compute primitives should consume tile fragments with precision metadata, but M5 should execute only 32-bit forms.

M5 behavior:

```text
f32 x f32 -> f32 accumulation/result
possibly i32/u32 arithmetic where already supported
no f16/u8/i8/fp8/int4/fp4 support
```

Future behavior:

```text
f16 storage -> f32 carrier compute -> f32/f16 result policy
u8/i8 quantized storage -> dequantize/sign-extend/zero-point adjust -> f32/i32 carrier compute
fp8/fp4/int4 storage -> software decode + f32/i32 compute unless a hardware-specific optimized path exists
optional packed-byte ALU paths for specialized operations, not generic tensor-core matmul
```

The key design requirement is that `tile_contract`/`tile_dot` should interface both upward and downward:

```text
Upward:
  It must be a natural target for Triton dot-like operations and IREE Linalg matmul/contract/reduction patterns.

Downward:
  It must canonicalize to VC4Tile core loads/copies/arithmetic/reductions and eventually SSAVC4 pack/unpack/VPM/VDW operations.
```

### 7.5 Reductions

Reductions need precision metadata because accumulation type matters.

M5 behavior:

```text
32-bit f32/i32 reductions only
```

Future behavior:

```text
low-precision storage may be widened before reduction
accumulator type must be explicit
quantized reductions need scale/zero-point rules or diagnostics
```

---

## 8. Development timeline

### Stage A: M5 ergonomic VC4Tile, 32-bit only

Purpose:

```text
Make VC4Tile ergonomic and CuTe/TK-like enough to be useful as a common future target.
Do not implement Triton/IREE producer lowering.
Do not implement sub-32 executable precision.
```

Precision-related tasks:

1. Add precision metadata categories to surface op definitions and design docs.
2. Make defaults exactly match 32-bit behavior.
3. Add verifier/canonicalization checks that reject unsupported non-32 precision.
4. Add tests proving:
   * 32-bit defaults roundtrip;
   * 32-bit explicit metadata canonicalizes correctly;
   * f16/u8/i8/fp8/int4/fp4 fields are rejected;
   * quantization metadata is rejected unless `none`;
   * unsupported precision diagnostics are deterministic.
5. Ensure no lower-half changes accidentally fake precision support.

Expected executable support at end of M5:

```text
32-bit ergonomic VC4Tile works on real hardware.
Precision metadata exists but only default/exact_32 executes.
```

### Stage B: Triton v1, 32-bit only

Purpose:

```text
Lower substantive Triton kernels into ergonomic/core VC4Tile.
Prove end-to-end 32-bit Triton -> VC4Tile -> SSAVC4 -> VC4 -> hardware.
```

Precision-related tasks:

1. Accept only 32-bit producer types/forms covered by the spec.
2. Reject Triton `tl.dot` inputs outside the supported 32-bit set.
3. Reject `tl.dot_scaled`, fp8, fp4, f16, bf16, int8, int4, and microscaling unless a specific 32-bit fallback is deliberately specified.
4. Map supported dot/load/store/reduce forms to VC4Tile fields with `precision_policy = exact_32`.
5. Add diagnostics that name the unsupported Triton precision/format.

Expected support:

```text
Triton v1 end-to-end works for selected 32-bit kernels.
Sub-32 producer forms fail clearly.
```

### Stage C: IREE/JAX/PyTorch v1, 32-bit only

Purpose:

```text
Lower selected IREE dispatch/Linalg paths into VC4Tile.
Prove end-to-end 32-bit JAX/PyTorch via IREE -> VC4Tile -> hardware.
```

Precision-related tasks:

1. Accept only 32-bit Linalg/dispatch cases.
2. Reject quantized types and dequantize/quantize patterns not implemented.
3. Reject fp16/bf16/fp8/int8/int4 low-precision cases.
4. Preserve/diagnose JAX precision hints and preferred accumulator types.
5. Add tests for unsupported StableHLO/Quant/IREE/PyTorch quantized patterns.

Expected support:

```text
IREE/JAX/PyTorch v1 end-to-end works for selected 32-bit kernels.
Quantized/sub-32 producer paths fail clearly.
```

### Stage D: Precision milestone P1 — sub-32 storage and copy planning

Purpose:

```text
Implement the first real sub-32 hardware support where VC4 is strongest: storage, VPM, VDW/VDR, and pack/unpack around 32-bit carriers.
```

Recommended first supported forms:

```text
f16 storage -> f32 carrier compute
u8 storage -> u32/i32/f32 carrier compute
possibly i8 storage -> i32/f32 carrier compute via explicit sign/zero-point handling
```

Required compiler work:

* VC4Tile surface supports selected non-32 storage metadata.
* Canonicalization emits explicit core conversion/copy intent.
* Copy planner chooses VPM packed/laned and VDW/VDR modes.
* SSAVC4 gains machine-level pack/unpack/VPM/VDW attrs/ops as needed.
* Scheduled VC4 emits exact setup words and pack/unpack bits.
* Hardware fixtures validate device copyback against CPU references.

Representative hardware fixtures:

```text
f16 global/storage -> f32 compute -> f32 store
f16 global/storage -> f32 compute -> f16 store, if store packing is implemented
u8 global/storage -> f32 dequantize -> f32 store
u8 VPM packed copy -> f32/u32 carrier compute -> 32-bit store
16-bit VPM laned copy -> f32 carrier compute
VDW packed 8-bit store to global memory
VDW packed 16-bit store to global memory
copy_tile global<->VPM packed/laned shape tests
negative tests for unsupported fp8/int4/fp4 fast paths
```

### Stage E: Precision milestone P2 — quantization semantics

Purpose:

```text
Support explicit quantize/dequantize around selected storage forms.
```

Recommended support order:

1. per-tensor symmetric u8/i8-like dequantization;
2. per-row/per-channel scale for simple matmul/GEMV patterns;
3. affine scale + zero-point;
4. blockwise/sub-channel quantization if needed by producer inputs.

Required compiler work:

* Scale and zero-point operands/attrs become executable.
* CPU-reference fixtures check numerical tolerance and exact integer behavior where applicable.
* Producer diagnostics become support checks instead of blanket rejection for the implemented subset.

### Stage F: Precision milestone P3 — compute primitive integration

Purpose:

```text
Let tile_contract/tile_dot/reductions consume precision metadata in real kernels.
```

Possible supported forms:

```text
f16 storage, f32 accumulation
u8/i8 storage, explicit dequantize, f32 accumulation
int-like storage, i32 accumulation for simple dot/GEMV if profitable/correct
specialized v8 ALU epilogues or image/byte kernels
```

Policy:

```text
Do not claim native fp8/int4/int8 tensor-core-like behavior.
Implement only what CPU-reference and hardware fixtures prove.
```

### Stage G: Advanced storage formats, optional

Purpose:

```text
Support fp8/fp4/int4/MX-like formats as storage/software-emulated paths if producer demand justifies it.
```

Expected behavior:

```text
fp8/fp4/int4 storage decode in software to 32-bit carriers
scales applied explicitly
performance likely memory-saving/correctness-oriented, not tensor-core-like
```

---

## 9. Required M5 precision-field behavior

M5 should provide a minimal but strict precision skeleton.

### 9.1 Supported values

```text
storage_type:
  default, f32, i32, u32

expressed_type:
  default, f32, i32, u32

accumulator_type:
  default, f32, i32, u32

precision_policy:
  default, exact_32

quantization:
  default, none

scale_granularity:
  default, none

zero_point_policy:
  default, none

packing:
  default, none, native_32
```

### 9.2 Rejected values

```text
f16
bf16
u16/i16 if not already explicitly supported as 32-bit-like storage
u8/i8
fp8_e4m3/fp8_e5m2
fp4/int4/intx
mxfp4/mxfp8/nvfp4
non-none quantization
scale operands
zero-point operands
packed/laned sub-32 VPM modes
sub-32 VDW/VDR modes
```

### 9.3 Required diagnostic shape

Diagnostics should be specific enough for future producer-lowering agents. Example:

```text
error: vc4tile.tile_load storage_type=f16 is not supported in M5; M5 executable precision is 32-bit only. Preserve this metadata only in non-executable planning tests or implement the post-M5 precision milestone before lowering this form.
```

or:

```text
error: vc4tile.tile_contract precision_policy=block_scaled is not supported in the 32-bit-only producer path; Triton tl.dot_scaled/fp4/fp8 lowering requires the later VC4 precision/quantization milestone.
```

---

## 10. Verification policy

Hardware remains the gold standard. For M5 itself, hardware fixtures should prove the 32-bit ergonomic ops. Precision-specific tests in M5 should mostly be roundtrip/canonicalization/diagnostic tests, because non-32 execution is intentionally unsupported.

### 10.1 M5 tests

Required M5 precision-related tests:

```text
Dialect/roundtrip:
  - default precision fields print/parse or are elided canonically
  - explicit exact_32 fields roundtrip if exposed
  - future fields rejected deterministically when executable

Canonicalization:
  - surface precision defaults normalize into core exact_32 behavior
  - unsupported sub-32 precision fails before SSAVC4 lowering

Conversion:
  - --convert-vc4tile-to-ssavc4 accepts only 32-bit canonicalized core
  - raw unsupported precision does not reach SSAVC4 silently

CodeGen/hardware:
  - 32-bit ergonomic kernels run on real hardware
  - no hardware fixture claims sub-32 support in M5
```

### 10.2 Producer v1 tests

Triton/IREE v1 tests should include unsupported precision diagnostics:

```text
Triton:
  - reject tl.dot with int8/fp8/f16/bf16 when not implemented
  - reject tl.dot_scaled with fp4/fp8 scales
  - accept selected f32/i32 forms

IREE/JAX/PyTorch:
  - reject quantized tensors/types/patterns
  - reject dequantize+matmul patterns until precision milestone
  - reject unsupported precision hints/preferred accumulation types
  - accept selected 32-bit Linalg/dispatch forms
```

### 10.3 Later precision hardware tests

Precision milestones must add real hardware tests with device-derived results. No fixture may pass by printing synthetic `VC4_TEST_RESULT`. Every precision fixture needs:

```text
candidate-generated path only
fresh generation by default
device allocation/copy-in/copy-out
CPU reference comparison
status derived from mismatch counts / launch failure counts
expected.json checking actual fields
artifact/scheduled checks for exact pack/unpack/VPM/VDW evidence when relevant
```

---

## 11. Risks and mitigations

### Risk: metadata fields become decorative and later ignored

Mitigation:

```text
M5 must reject unsupported non-default values.
Producer v1 must reject unsupported upstream precision forms.
No silent widening.
```

### Risk: VC4Tile becomes too hardware-specific for producer lowering

Mitigation:

```text
Surface fields should describe semantic precision, not raw pack bits.
Raw VC4 pack/VPM/VDW modes appear lower, in SSAVC4/scheduled VC4.
```

### Risk: VC4Tile becomes too generic and loses copy-planner power

Mitigation:

```text
Surface and core should retain layout, role, storage_type, and packing intent.
The copy planner must see enough information to choose VPM/VDW/TMU modes later.
```

### Risk: fp8/int4 support is overpromised

Mitigation:

```text
Document that VC4 lacks native tensor-core-like fp8/fp4/int4 matrix compute.
Treat fp8/fp4/int4 as storage/software-emulated until proven otherwise.
Use diagnostics and CPU-reference fixtures.
```

### Risk: sub-32 implementation touches every level at once

Mitigation:

```text
Sequence precision milestones vertically but narrowly:
P1 storage/copy; P2 quantization; P3 compute primitive integration; P4 advanced formats.
Each slice must include dialect, canonicalization, lowering, scheduled artifact, and hardware proof for the feature it claims.
```

---

## 12. Final locked decision

The locked decision is:

```text
M5 is ergonomic VC4Tile only.
M5 executable semantics are 32-bit only.
M5 must add forward-looking precision metadata/fields where appropriate.
M5 must reject unsupported sub-32/quantized fields deterministically.
Triton v1 is 32-bit only.
IREE/JAX/PyTorch v1 is 32-bit only.
Sub-32 precision is implemented only after those end-to-end paths work.
The first sub-32 work should target storage/copy/pack/unpack, not native low-precision tensor compute.
```

This is the cleanest path because it preserves an implementable route to end-to-end ML kernels while keeping the IR architecture ready for real mixed-precision and quantization work. The compiler should not try to look like modern tensor-core hardware where VC4 cannot support that model natively. It should instead expose modern precision semantics at the VC4Tile surface, lower them through a VC4-aware copy planner and explicit conversion model, and implement only the hardware-backed subsets that real QPU tests can prove.

---

## 13. Source notes

**[VC4-ARG]** Broadcom, *VideoCore IV 3D Architecture Reference Guide*, user-provided PDF, September 16, 2013. Relevant sections: QPU architecture and ALU model; pack/unpack bits and tables; VPM/VCD/VDW QPU read/write and setup formats; QPU instruction set and packed-byte ALU operations.

**[Triton-dot]** Triton official documentation, `triton.language.dot`: https://triton-lang.org/main/python-api/generated/triton.language.dot.html

**[Triton-dot-scaled]** Triton official documentation, `triton.language.dot_scaled`: https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html

**[Triton-block-scaled]** Triton official tutorial, “Block Scaled Matrix Multiplication”: https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html

**[IREE-Linalg]** IREE blog/tutorial, “IREE / MLIR / Linalg tutorial”: https://iree.dev/community/blog/2024-01-29-iree-mlir-linalg-tutorial/

**[IREE-passes]** IREE pass reference, GlobalOptimization: https://iree.dev/reference/mlir-passes/GlobalOptimization/

**[StableHLO-quant]** OpenXLA StableHLO Quantization documentation: https://openxla.org/stablehlo/quantization

**[MLIR-Quant]** MLIR Quant dialect documentation: https://mlir.llvm.org/docs/Dialects/QuantDialect/

**[TorchAO]** PyTorch torchao Quantized Inference documentation: https://docs.pytorch.org/ao/stable/workflows/inference.html

**[JAX-matmul]** JAX documentation, `jax.numpy.matmul`: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.matmul.html

**[JAX-precision]** JAX documentation, `jax.default_matmul_precision`: https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html
