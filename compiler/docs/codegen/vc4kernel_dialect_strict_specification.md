# VC4Kernel Dialect Strict Specification

**Status:** corrected Stage 1 target specification plus Surface v2 pre-vector lock for the `vc4kernel` dialect and its lowering boundary.
**Date:** 2026-06-06.
**Audience:** implementation agents maintaining `vc4kernel`, `VC4KernelToSSAVC4`, SSAVC4, scheduled `vc4`, `vc4-codegen`, and the libpi-backed launch/runtime path.  
**Primary purpose:** replace the earlier strict specification with the corrected final contract after the pre-lowering design review.

---

## 0. Normative rule

This document is the source of truth for the current `vc4kernel` Stage 1 contract.

The implementation must reflect the clean final stack we want. Existing SSAVC4, scheduled `vc4`, `vc4-codegen`, and runtime behavior is implementation material, not a constraint. If an existing lower-half name, schema, verifier rule, or runtime field is unnatural, ambiguous, or only historically convenient, update the lower half instead of contorting `vc4kernel` to match it.

Anything not explicitly permitted by this document is forbidden in verified `vc4kernel` IR.

The required final stack remains:

```text
Triton-emitted TTIR / future producer IR
  -> standard MLIR value layer
     vector + memref + arith + math + scf/cf
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> vc4-codegen artifacts
  -> libpi-backed VC4 runtime
  -> real VC4 hardware
```

The implementation order is bottom-up:

```text
1. Correct and re-lock vc4kernel according to this document.
2. Update SSAVC4 / scheduled vc4 / codegen / runtime to match this document.
3. Prove vc4kernel -> ssavc4 -> scheduled vc4 -> hardware.
4. Only then specify and implement the upstream standard vector surface.
5. Only then lower real Triton-emitted TTIR to the standard value layer.
```

This document deliberately supersedes the previous strict spec in the following important ways:

```text
- remove the lane-id operation
- remove the block-id operation
- remove user-authored vc4kernel resource summary dictionaries
- keep vc4kernel.fragment_cmp
- broaden !vc4kernel.pred<16> from only normalizable tails to a true 16-lane predicate/mask with structured and general classes
- define resource requirements as compiler-computed metadata emitted below vc4kernel
- define runtime-assigned vpm_base_row and semaphore_base builtins
- require hardware-derived VPM/VDR/VDW mode schema in the lower half
- support 32-bit horizontal, vertical, and strided/pitched VPM/VDR/VDW movement in v1
- keep sub-32 packed/laned schema only as non-executable metadata until the sub-32 precision milestone
```

---

## 1. Design principles

Every design and implementation decision must be evaluated in this order:

```text
1. Upstream expressibility:
   What must the future standard vector layer and Triton TTIR path be able to express?

2. VC4 hardware truth:
   What is the natural, legitimate mapping onto QPU SIMD, uniforms, TMU, VPM, VDR/VCD, VDW, semaphores, and scheduled QASM?

3. Triton/NVIDIA precedent:
   Where the NVIDIA Triton backend has an analogous decision, mirror the level of intent rather than copying names mechanically.
```

### 1.1 NVIDIA/Triton resource precedent

The relevant NVIDIA/Triton pattern is separation of concerns:

```text
source/autotune configuration:
  num_warps
  num_stages
  num_ctas
  maxnreg

compiler-planned target IR metadata:
  total number of warps after target planning / specialization
  shared-memory byte requirement
  tensor-memory requirement on newer NVIDIA targets
  scratch-memory size/alignment
  final entry name and target codegen metadata

runtime launch metadata:
  compact metadata needed to launch the kernel and check resource limits
```

Triton does not make source IR manually maintain a redundant summary like `uses_shared=true, shared_bytes=N`. It has explicit local/shared-memory IR concepts, target passes compute allocation offsets and total shared-memory usage, and the final NVIDIA backend packs compact launch metadata such as `num_warps`, `num_ctas`, and `shared`.

The VC4 equivalent is:

```text
vc4kernel source:
  schedule mode
  warps_per_block
  formal ABI args
  explicit VPM/dataflow ops
  fragment/predicate/memory operations

vc4kernel verifier/planner and VC4KernelToSSAVC4:
  compute user VPM rows
  compute hidden compiler staging rows
  compute TMU/VPM/VDR/VDW/barrier/semaphore usage
  assign per-block and per-warp resource requirements

SSAVC4 / scheduled vc4 / manifest:
  carry computed launch ABI and resource metadata

runtime:
  assign vpm_base_row and semaphore_base per resident logical request/block
  pack builtins and launch QPU work
```

---

## 2. Core identity of the dialect

### 2.1 What `vc4kernel` is

`vc4kernel` is the VC4 target-kernel planning dialect immediately above SSAVC4.

It represents a kernel after generic value-layer semantics have been lowered into VC4-specific execution fragments, predicates, memory paths, launch identity, VPM resources, and synchronization requirements.

It answers:

> How should this already-tiled, already-vectorized kernel execute on VC4 QPUs, VPM, TMU, VDR/VCD, VDW, uniforms, and semaphores?

### 2.2 What `vc4kernel` is not

`vc4kernel` is not:

```text
- a producer-facing tile DSL
- a CuTe clone
- a ThunderKittens clone
- a Triton clone
- a replacement for the MLIR vector dialect
- a replacement for memref/arith/scf/cf in the generic value layer
- SSAVC4 machine SSA
- scheduled vc4 / QASM-near IR
- a host/runtime ABI dialect
- a sub-32 precision dialect
- a compatibility layer for old vc4tile
```

### 2.3 Required layer separation

```text
standard value layer:
  generic fixed-size value semantics
  vector.transfer_read/write
  vector.contract
  vector.reduction
  vector.mask
  vector.transpose
  memref logical memory
  arith/math/scf/cf

vc4kernel:
  VC4 execution-plan semantics
  16-lane fragments
  VC4 predicates/masks
  raw device-pointer words
  TMU/VDR/VPM/VDW path choice
  explicit VPM allocation and access
  compiler-managed VPM staging requirements
  program/warp/lane-range identity
  schedule mode and warps_per_block
  barriers
  target fragment arithmetic/reductions
  scalar core CFG

ssavc4:
  target machine SSA
  uniforms
  element_number
  ALU dataflow
  condition plans / flags / branches
  TMU requests/reads
  VPM reads/writes
  VDR loads
  VDW stores
  semantic launch/resource metadata
  semaphores/barriers
  thread_end

scheduled vc4:
  register-allocated, scheduled, QASM-near machine operations
```

No `vc4kernel` pass may lower directly to scheduled `vc4`. The only valid lower path is:

```text
vc4kernel -> ssavc4 -> scheduled vc4 -> artifacts/runtime/hardware
```

---

## 2.4 VC4Kernel Surface v2 pre-vector lock

Surface v2 is the pre-vector/pre-Triton hardening lock for `vc4kernel`. It preserves the existing accepted dynamic rectangular, spill, runtime GEMV/GEMM, lower-half branch-layout, and hardware fixture baseline while planning the next surface changes. It does not authorize new execution semantics by documentation alone.

### Normative boundary

```text
standard value layer:
  owns vector, memref, arith, math, scf, and cf producer/value semantics.

vc4kernel:
  owns VC4 target execution-plan semantics above SSAVC4.

lower path:
  vc4kernel -> ssavc4 -> scheduled vc4 -> artifacts/runtime/hardware.
```

There is no direct `VC4KernelToVC4` path. `vc4kernel` is not a producer tile DSL, not a replacement for the standard value layer, and not a compatibility layer for old VC4Tile concepts.

### No compatibility or special-case bias

Current special-case operations are migration targets, not compatibility promises:

```text
fragment_add / fragment_sub / fragment_mul / fragment_shl:
  historical legacy migration target removed_in_p1 after replacement by the P1 general hardware-faithful ALU surface.

fragment_reduce with add only:
  migration target for P4 general reductions.

tmu_load_fragment implicit safe-address behavior:
  migration target for P7 explicit safe inactive-load policy.

vdw_store_fragment implicit inactive-store behavior:
  migration target for P8 explicit inactive-store policy.
```

Future agents must not keep historical legacy migration ops `fragment_add`, `fragment_sub`, `fragment_mul`, `fragment_shl`, add-only reduction specialness, or implicit TMU safe-address semantics merely because they already existed; the fragment arithmetic spellings are removed_in_p1.

### Locked P0-P13 phase order

```text
P0  spec/matrix/audit
P1  general ALU
P2  bitcast/constants
P3  comparisons
P4  reductions
P5  scalar arith
P6  memory/coherency
P7  TMU safe inactive load
P8  VDW inactive store v1
P9  pack/unpack/subword
P10 SFU/fastmath
P11 dynamic rotate/shuffle
P12 dynamic VPM/VDR/VDW coordinates
P13 final support matrix/pre-vector lock

Deferred:
  arbitrary sparse VDW stores
```

P1-P13 are planned phases. This document does not claim those phases are implemented until their verifier, conversion, lower-half, and hardware or deterministic-reject evidence exists.

### Locked feature policies

```text
P1 general ALU:
  define a general hardware-faithful ALU surface instead of preserving one-off fragment ALU ops as the long-term API.

P2 bitcast/constants:
  define bit-level carrier reinterpretation and constant materialization without admitting producer dialect operations.

P3 comparisons:
  define general comparison semantics and predicate production beyond current special-case comparisons.

P4 reductions:
  define a general reduction family instead of add-only specialness.

P5 scalar arith:
  define the scalar arithmetic subset that belongs in vc4kernel planning, distinct from producer-level arith ownership.

P6 memory/coherency:
  lock TMU, VDR, VDW, VPM, spill, and inactive-lane coherency rules.

P7 TMU safe inactive load:
  replace implicit safe-address behavior with an explicit safe offset / inactive-load policy.

P8 VDW inactive store v1:
  support full, tail, and rectangular preserve semantics; arbitrary sparse masks deterministic-reject.

P9 pack/unpack/subword:
  admit only hardware-proven pack/unpack and sub-32 VPM mode semantics, with executable rejects until proven.

P10 SFU/fastmath:
  add SFU-derived math only behind an explicit fastmath/approx contract.

P11 dynamic rotate/shuffle:
  admit dynamic rotate/shuffle only if the downstream and hardware proof exists.

P12 dynamic VPM/VDR/VDW coordinates:
  admit dynamic coordinates/pitch only where lower-half and hardware proof exists.

P13 final matrix:
  lock each feature to verifier, VC4KernelToSSAVC4, lower-half, and hardware or deterministic-reject proof.
```

### Fast-math and SFU policy

Default math is exact/conservative. Approximate SFU-derived math is opt-in only through an explicit fastmath/approx contract. `sqrt` may lower through `rsqrt` only under that explicit contract unless an exact sequence is implemented and tested. NaN, Inf, and signed-zero exactness must not be promised without explicit tests covering those cases.

### VDW masked-store policy

VDW v1 supports:

```text
full store:
  all lanes/elements active.

tail store:
  contiguous prefix active, inactive destination lanes preserved.

rectangular store:
  active rows/columns stored, inactive destination region preserved.
```

Arbitrary sparse VDW masks deterministic-reject until a later hardware-proven phase. The compiler must not silently decompose sparse stores into read-modify-write stores unless that deferred phase exists and has verifier, lowering, lower-half, and hardware proof.

### Memory path and coherency policy

The legal memory paths are explicit:

```text
TMU:
  global memory to register fragment read path.

VDR:
  global memory to VPM path.

VDW:
  VPM or compiler-staged register fragment to global memory path.
```

Compiler spill slots written by VDW must reload through the coherent VDR->VPM path, not TMU, unless a future architecture-backed invalidation/coherency mechanism is specified and proven.

### Surface matrix requirement

Every Surface v2 feature must have:

```text
verifier coverage
VC4KernelToSSAVC4 coverage
lower-half coverage where needed
hardware fixture proof or deterministic-reject proof
```

### Forbidden permanent surface

The following are not permanent `vc4kernel` surface features:

```text
tile_broadcast
tile_dot
tile_matmul
tile_contract
fragment_contract
producer-level layout algebra
arbitrary sparse VDW store before its deferred phase
integer div/mod unless a library sequence is designed
atomics
tile-buffer color/Z/stencil
texture filtering
cube maps
varyings for compute v1
```

---

## 3. Names, files, and pass names

### 3.1 Dialect identity

```text
MLIR operation prefix: vc4kernel.
MLIR type prefix:      !vc4kernel.*
MLIR attribute prefix: #vc4kernel.*
C++ namespace:         mlir::vc4kernel
TableGen prefix:       VC4Kernel
Dialect class:         VC4KernelDialect
```

### 3.2 Expected file layout

```text
compiler/include/vc4/Dialect/VC4Kernel/IR/VC4KernelDialect.td
compiler/include/vc4/Dialect/VC4Kernel/IR/VC4KernelAttrs.td
compiler/include/vc4/Dialect/VC4Kernel/IR/VC4KernelTypes.td
compiler/include/vc4/Dialect/VC4Kernel/IR/VC4KernelOps.td
compiler/include/vc4/Dialect/VC4Kernel/IR/VC4KernelDialect.h
compiler/include/vc4/Dialect/VC4Kernel/IR/VC4KernelOps.h
compiler/include/vc4/Dialect/VC4Kernel/IR/VC4KernelTypes.h
compiler/include/vc4/Dialect/VC4Kernel/IR/VC4KernelAttrs.h
compiler/lib/Dialect/VC4Kernel/IR/VC4KernelDialect.cpp
compiler/lib/Dialect/VC4Kernel/IR/VC4KernelOps.cpp
compiler/lib/Dialect/VC4Kernel/IR/VC4KernelTypes.cpp
compiler/lib/Dialect/VC4Kernel/IR/VC4KernelAttrs.cpp
compiler/include/vc4/Conversion/VC4KernelToSSAVC4/VC4KernelToSSAVC4.h
compiler/lib/Conversion/VC4KernelToSSAVC4/VC4KernelToSSAVC4.cpp
compiler/test/Dialect/VC4Kernel/...
compiler/test/Conversion/VC4KernelToSSAVC4/...
compiler/test/CodeGen/VC4Kernel/...
```

### 3.3 Required passes

```text
--verify-vc4kernel
--convert-vc4kernel-to-ssavc4
```

Optional helper passes are allowed only if they remain inside the `vc4kernel` boundary and do not admit producer dialects or direct scheduled-VC4 lowering. A helper such as `--legalize-vc4kernel-cfg` may lower small CFG details inside `vc4kernel`; it must not accept raw `scf.*` as verified core.

---

## 4. Legal type system

### 4.1 Legal scalar types

Only these builtin scalar types are legal inside verified `vc4kernel` IR:

```text
i1
i32
f32
```

Interpretation:

```text
i1   scalar condition value used for scalar CFG/select; lowers through condition plans/flags, not normally as generic data
i32  signed or unsigned 32-bit integer / raw device pointer word / byte offset / index / resource count
f32  32-bit floating-point scalar, including scalar f32 uniforms
```

`u32` is represented as MLIR `i32` with unsigned interpretation documented by operation semantics or ABI metadata. There is no distinct unsigned builtin type.

### 4.2 Legal fragment carrier types

Only these vector carrier types are legal:

```text
vector<16xi32>
vector<16xf32>
```

These are data carriers for one VC4 QPU 16-lane fragment. They are legal types even though `vector.*` dialect operations are forbidden inside verified `vc4kernel` IR.

### 4.3 Legal custom types

```text
!vc4kernel.pred<16>
!vc4kernel.vpm_tile
```

No other `vc4kernel` types are legal unless this document is revised.

### 4.4 Predicate type

`!vc4kernel.pred<16>` is the only legal predicate type.

It is not equivalent to `vector<16xi1>`.

It represents a 16-lane VC4-planned predicate/mask over the current fragment. It can represent structured masks and general masks.

Required predicate classes:

```text
full:
  all 16 lanes active

empty:
  no lanes active

tail_prefix:
  active lanes are a contiguous prefix [0, active_count), 0 <= active_count <= 16

rect_row:
  the predicate for one row-fragment of a rectangular tile; semantically row-in-bounds AND tail_prefix on columns

general_mask:
  arbitrary per-lane predicate, including fragment_cmp results and Boolean compositions that do not simplify to a structured class
```

Consumers must declare which classes they accept and what fallback they use for general masks. The verifier must reject only masks that a consumer cannot lower correctly.

### 4.5 VPM handle type

`!vc4kernel.vpm_tile` is an opaque handle to a statically planned VPM allocation within the logical kernel block/request. It is a VC4 resource handle. It is not a tensor, memref, layout algebra object, generic tile descriptor, pointer, or runtime heap allocation.

### 4.6 Forbidden types

All other types are forbidden inside verified `vc4kernel` IR, including:

```text
index
i8 / i16 / i64
f16 / bf16 / f64
fp8 / fp4 / int4 / uint4 / other sub-32 executable types
vector<16xi1>
vector widths other than 16
vector element types other than i32/f32
memref
tensor
Triton pointer types
Triton block/tensor types
LLVM pointer types
SSAVC4 async token types
scheduled VC4 dialect types
```

Sub-32 data movement modes may be modeled in lower-half attrs as schema for future work, but no executable sub-32 `vc4kernel` value type or lowering is legal in this stage.

---

## 5. Legal dialects and operation families inside verified `vc4kernel`

### 5.1 Allowed dialects

Verified `vc4kernel` IR may contain operations from only these dialects:

```text
vc4kernel
arith, restricted to scalar operations listed below
cf, restricted to cf.br and cf.cond_br
builtin/module wrappers as required by MLIR
```

### 5.2 Allowed scalar `arith` operations

```text
arith.constant
arith.addi
arith.subi
arith.muli
arith.shli
arith.cmpi
arith.select
```

Restrictions:

```text
- arith operations may not have vector result types.
- arith operations may not have vector operands.
- arith.constant may produce only scalar i1/i32/f32.
- arith.addi/subi/muli/shli may operate only on scalar i32.
- arith.cmpi may compare only scalar i32 values and produce scalar i1.
- arith.select condition must be scalar i1 and selected values must be scalar i1/i32/f32.
```

Scalar `i1` lowers through condition plans. It is not assumed to be a persistent hardware flag value.

### 5.3 Allowed `cf` operations

```text
cf.br
cf.cond_br
```

Restrictions:

```text
- cf.cond_br condition must be scalar i1.
- cf successor operands may use only legal vc4kernel types.
- cf terminators are legal only inside vc4kernel.kernel regions.
- raw scf must not appear in verified vc4kernel IR.
- !vc4kernel.vpm_tile must not be passed through block arguments unless the verifier proves it is the same static allocation handle; v1 should conservatively reject VPM handle block args.
```

### 5.4 Forbidden dialects

Always forbidden inside verified `vc4kernel` IR:

```text
vector
memref
tensor
scf
func
linalg
gpu
tt
ttg
nvgpu
nvvm
rocdl
spirv
iree
stablehlo
mhlo
llvm
ssavc4
vc4
```

Important: `vector<16xi32>` and `vector<16xf32>` types are legal. `vector.*` operations are not.

---

## 6. Kernel operation, launch ABI, and computed resources

### 6.1 `vc4kernel.kernel`

Summary:

```text
Top-level VC4 target-kernel planning operation.
```

Required traits/interfaces:

```text
IsolatedFromAbove
Symbol
FunctionOpInterface-compatible function-like region
No results
```

Final syntax target:

```mlir
vc4kernel.kernel @name(%arg0: i32, %arg1: f32) attributes {
  public_name = "name",
  schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
  warps_per_block = 1 : i32,
  arg_attrs = [ ... ]
} {
^entry(%arg0: i32, %arg1: f32):
  ...
  vc4kernel.return
}
```

There is no user-authored `resource = { ... }` dictionary in source `vc4kernel`.

### 6.2 Formal arguments

Formal arguments are user/caller-provided values. They lower to `vc4.launch_abi.args[]` on the generated SSAVC4 function and then to sequential uniform reads.

Legal formal argument types:

```text
i32
f32
```

`i32` may represent:

```text
- by-value integer scalar
- raw vc4_deviceptr_t device pointer word
- by-value unsigned scalar
```

`f32` represents a by-value 32-bit float scalar. It lowers to a scalar `f32` uniform read. If the current lower half lacks scalar f32 uniform support, the lower half must be updated.

No formal argument may be a memref, tensor, vector, predicate, VPM handle, index, pointer type, or sub-32 type.

### 6.3 Argument metadata

`vc4kernel.kernel` must carry `arg_attrs`, one entry per formal argument.

Each entry must be a dictionary with these fields:

```text
name:        string, required
kind:        "scalar" | "buffer", required
direction:   "by_value" | "in" | "out" | "inout", required
type:        "u32" | "i32" | "f32", required for scalar
elem_type:   "u32" | "i32" | "f32", required for buffer
```

Rules:

```text
- kind="buffer" means the formal argument type is i32 and the value is one raw VC4 device pointer word.
- No full host-side memref descriptor is represented in vc4kernel.
- uniform_index is not user-authored in vc4kernel source. It is assigned during vc4kernel -> ssavc4 launch ABI construction.
- The verifier must reject missing, malformed, duplicate, or length-mismatched arg_attrs.
```

### 6.4 Schedule mode and warps_per_block

`vc4kernel.kernel` must carry:

```text
schedule_mode = #vc4kernel.schedule_mode<independent_vector>
```

or:

```text
schedule_mode = #vc4kernel.schedule_mode<cooperative_block>
```

and:

```text
warps_per_block = N : i32
```

Rules:

```text
independent_vector:
  warps_per_block must be 1.
  Each logical request/block is one QPU warp executing one 16-lane SIMD fragment at a time.
  Barriers are forbidden.
  VPM use is legal, but VPM rows are private to the resident logical request and addressed relative to runtime-assigned vpm_base_row.

cooperative_block:
  warps_per_block must be in [1, 12].
  All warps in one logical block are resident together.
  All warps in one block share the same vpm_base_row and semaphore_base.
  warp_id is meaningful and ranges 0 <= warp_id < warps_per_block.
  Barriers are legal only in cooperative_block mode.
```

`require_full_block_residency` is not a user-authored source attr. It is derived from cooperative scheduling and barrier/resource planning below `vc4kernel`.

### 6.5 Computed resource metadata

The compiler must compute a semantic resource summary during `vc4kernel -> ssavc4`. This summary is emitted as lower-half metadata on the resulting SSAVC4 function and propagated into scheduled VC4 / manifest / runtime.

Recommended lower-half attr name:

```text
vc4.resource
```

Required computed fields:

```text
schedule_mode:                         "independent_vector" | "cooperative_block"
warps_per_block:                       i32
qpu_slots_per_block:                   i32, equal to warps_per_block for v1

uses_tmu:                              bool
uses_vpm:                              bool
uses_vpm_qpu_read:                     bool
uses_vpm_qpu_write:                    bool
uses_vdr:                              bool
uses_vdw:                              bool
uses_barrier:                          bool

user_vpm_rows_per_block                i32
compiler_vpm_staging_rows_per_warp:    i32
compiler_vpm_staging_rows_per_block:   i32
total_vpm_rows_per_block               i32
total_vpm_bytes_per_block:             i32, derived = total_vpm_rows_per_block * 16 * 4

semaphore_count_per_block:             i32
requires_vpm_base_row_builtin:         bool
requires_semaphore_base_builtin:       bool
```

No old compatibility-only fields may be preserved unless they have a real semantic meaning. In particular, ambiguous fields such as `shared_vpm_bytes`, `user_shared_vpm_rows_per_block`, and source-authored `uses_vpm`/`uses_barrier` should be replaced by computed semantic fields.

### 6.6 Resource computation rules

```text
user_vpm_rows_per_block
  sum of explicit vc4kernel.vpm_alloc rows after deterministic allocation planning.

compiler_vpm_staging_rows_per_warp:
  rows needed for per-warp hidden staging, especially vdw_store_fragment register->global lowering.

compiler_vpm_staging_rows_per_block:
  rows needed for block-level hidden staging not owned by a single warp.

total_vpm_rows_per_block
  user_vpm_rows_per_block
  + warps_per_block * compiler_vpm_staging_rows_per_warp
  + compiler_vpm_staging_rows_per_block

uses_vpm:
  true if any explicit VPM op exists or hidden staging rows are required.

uses_vdr / uses_vdw / uses_tmu:
  derived from memory ops and lowerer-selected fallback paths.

semaphore_count_per_block:
  0 if no barrier.
  4 for the v1 cooperative barrier protocol unless a later verified protocol revises this.
```

Resource validity:

```text
- total_vpm_rows_per_block must be <= 64 for v1 general-purpose VPM window planning.
- total_vpm_bytes_per_block must be <= 4096 when the runtime reserves a 4 KiB general-purpose VPM window.
- semaphore_count_per_block must be <= 16.
- warps_per_block must be <= 12.
```

### 6.7 Runtime-assigned launch builtins

`vc4kernel` source does not contain these as formal args. They are inserted into `vc4.launch_abi.builtins[]` only when needed:

```text
program_id(axis):
  logical program/request index for launch-grid axis 0, 1, or 2. Required when vc4kernel.program_id is used.

num_programs(axis):
  logical launch-grid extent for axis 0, 1, or 2. Required when vc4kernel.num_programs is used.

warp_id:
  logical warp id within the cooperative block. Required when vc4kernel.warp_id is used or when barrier/lowering needs it.
  Constant 0 in independent_vector if needed internally, preferably omitted if not used.

warps_per_block:
  compile-time constant may be embedded; a builtin is required only if the lower-half barrier sequence expects it dynamically.

vpm_base_row:
  runtime-assigned base row for the current resident logical request/block. Required when total_vpm_rows_per_block > 0.

semaphore_base:
  runtime-assigned base semaphore id for the current resident cooperative block. Required when semaphore_count_per_block > 0.
```

The runtime residency planner computes:

```text
qpu_limit = floor(12 / warps_per_block)

vpm_limit =
  infinity, if total_vpm_rows_per_block == 0
  floor(64 / total_vpm_rows_per_block), otherwise

semaphore_limit =
  infinity, if semaphore_count_per_block == 0
  floor(16 / semaphore_count_per_block), otherwise

resident_blocks = min(qpu_limit, vpm_limit, semaphore_limit)
```

For each resident block/request slot:

```text
vpm_base_row = slot * total_vpm_rows_per_block
semaphore_base = slot * semaphore_count_per_block
```

Independent-vector kernels are treated as one-warp logical blocks for residency purposes.

---

## 7. Identity and lane operations

### 7.1 `vc4kernel.program_id`

Signature:

```mlir
%pid = vc4kernel.program_id {axis = 0 : i32} : i32
```

Semantics:

```text
Logical program/request index for launch-grid axis 0, 1, or 2.
program_id {axis = 0|1|2}
It is not physical QPU_NUMBER.
It is not a formal argument.
```

Axis values other than 0, 1, or 2 are invalid.

### 7.2 `vc4kernel.num_programs`

Signature:

```mlir
%n = vc4kernel.num_programs {axis = 0 : i32} : i32
```

Semantics:

```text
Logical launch-grid extent for launch-grid axis 0, 1, or 2.
num_programs {axis = 0|1|2}
It is not physical QPU identity.
It is not a formal argument.
```

Axis values other than 0, 1, or 2 are invalid.

### 7.3 `vc4kernel.warp_id`

Signature:

```mlir
%wid = vc4kernel.warp_id : i32
```

Semantics:

```text
Logical warp id within a cooperative block.
Range is 0 <= warp_id < warps_per_block.
It is not physical QPU_NUMBER.
Legal only in cooperative_block kernels or inside lowering-generated internal paths that need a constant/logical warp slot.
```

### 7.4 `vc4kernel.lane_range`

Signature:

```mlir
%lanes = vc4kernel.lane_range : vector<16xi32>
```

Semantics:

```text
Vector [0, 1, 2, ..., 15] in lane order.
Used to form fragment offsets and lane-wise values.
Lowers from SSAVC4 element_number / hardware ELEMENT_NUMBER.
```

### 7.5 Removed identity operations

These operations are not part of corrected `vc4kernel`:

```text
block-id operation
lane-id operation
vc4kernel.thread_id
```

Rationale:

```text
program_id is the logical Triton-facing program/block identity.
warp_id is the only needed cooperative intra-block identity.
lane_range is the SIMD-friendly lane expression.
lane_id and thread_id reintroduce scalar SIMT-shaped vocabulary and are not needed for the upstream vector/Triton path.
```

If a future pass needs a scalar element extracted from `lane_range`, that scalarization belongs above `vc4kernel` or in a separately specified later pass. It must not reintroduce `lane_id` as a core identity op.

---

## 8. Predicate model

### 8.1 Predicate design goal

`vc4kernel` predicates are target-planned 16-lane masks. They preserve enough structure to let the lowering choose correct VC4 implementations for:

```text
TMU per-lane loads
VDW contiguous full/tail/rect stores with preserve semantics
VDR full rectangular DMA
VPM read/write zero-fill semantics
fragment select/reduce
scalar pred.any / pred.all branches
```

### 8.2 Predicate classes

Every predicate value has one of these classes or can be conservatively classified as one of these classes during planning:

```text
full:
  all 16 lanes active

empty:
  no lanes active

tail_prefix:
  lanes [0, active_count) active, lanes [active_count, 16) inactive

rect_row:
  row-in-bounds AND tail-prefix column mask for one row-fragment

general_mask:
  arbitrary lane mask, including fragment_cmp results and Boolean compositions that do not simplify to a structured class
```

General masks are legal. They are not represented as `vector<16xi1>`; they are `!vc4kernel.pred<16>`.

### 8.3 Predicate operations

```mlir
%p = vc4kernel.pred.full : !vc4kernel.pred<16>
%p = vc4kernel.pred.empty : !vc4kernel.pred<16>
%p = vc4kernel.pred.tail %base, %limit : i32, i32 -> !vc4kernel.pred<16>
%p = vc4kernel.pred.rect %row, %rows, %col_base, %cols
  : i32, i32, i32, i32 -> !vc4kernel.pred<16>
%p = vc4kernel.pred.and %a, %b : !vc4kernel.pred<16>
%p = vc4kernel.pred.or  %a, %b : !vc4kernel.pred<16>
%p = vc4kernel.pred.not %a     : !vc4kernel.pred<16>
%b = vc4kernel.pred.any %p : !vc4kernel.pred<16> -> i1
%b = vc4kernel.pred.all %p : !vc4kernel.pred<16> -> i1
```

Boolean ops should canonicalize full/empty identities when possible, but non-simplified results are still legal as `general_mask` when consumers can lower them.

### 8.4 Consumer legality by predicate class

```text
fragment_select:
  accepts full, empty, tail_prefix, rect_row, general_mask

fragment_reduce:
  accepts all classes by first selecting inactive lanes to zero

pred.any / pred.all:
  accepts all classes; general_mask lowers through hardware flags and any/all branch conditions

tmu_load_fragment:
  accepts all classes; inactive lanes must return zero and must not issue unsafe out-of-bounds loads

vpm_write_fragment:
  accepts all classes; inactive lanes write zero into VPM

vpm_read_fragment:
  accepts all classes; inactive lanes return zero

vdw_store_fragment:
  accepts full, empty, tail_prefix, and rect_row for contiguous 32-bit row offsets with preserve semantics; arbitrary sparse general_mask deterministic-rejects until the deferred sparse VDW phase is hardware-proven

vdr_load_to_vpm:
  accepts only full static rectangular DMA semantics; runtime rectangular tails and leading dimensions use vdr_load_rect_to_vpm

vdw_store_vpm and vdw_store_vpm_fragment:
  direct DMA accepts full static rectangular or full/tail row semantics; runtime rectangular preserve-destination stores use vdw_store_rect_from_vpm; arbitrary sparse masks deterministic-reject
```

### 8.5 Forbidden predicate representations

Forbidden:

```text
vector<16xi1>
arith.cmpi producing vector<16xi1>
vector.mask
vector.create_mask
raw integer bitmasks
opaque predicate attributes on memory ops instead of !vc4kernel.pred<16> values
```

---

## 9. Fragment value operations

### 9.1 Fragment operation rule

All operations over `vector<16xi32>` or `vector<16xf32>` values inside `vc4kernel` must be `vc4kernel` operations. No `arith.*` operation may operate on vector types.

### 9.2 `vc4kernel.splat`

```mlir
%v = vc4kernel.splat %x : i32 -> vector<16xi32>
%v = vc4kernel.splat %x : f32 -> vector<16xf32>
```

Broadcasts a scalar to all 16 lanes. Scalar `f32` splat is first-class and must be supported by SSAVC4/lower-half uniform and splat machinery.

### 9.3 General fragment ALU ops

```mlir
%r = vc4kernel.fragment_alu.add %a, %b {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
%r = vc4kernel.fragment_alu.add %a, %b {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
%r = vc4kernel.fragment_alu.mul %a, %b {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
%r = vc4kernel.fragment_alu.mul %a, %b {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
```

Allowed carrier element types:

```text
i32
f32
```

The fragment ALU surface is hardware-faithful. The add-pipe and mul-pipe opcode attributes name the target VC4 ALU opcode. These ops do not promise producer-level generic arithmetic semantics.

Selected semantics:

```text
i32 add/sub/shl/logic/min/max:
  VC4 add-pipe opcode semantics over vector<16xi32> carriers.

i32 mul24:
  VC4 mul-pipe mul24 semantics only. It is not exact full-width i32 multiplication.

exact i32 multiply:
  Requires an explicit, tested expansion using hardware-faithful ops. It must never silently compile to mul24.

f32 add/sub/mul:
  VC4 f32 add-pipe or mul-pipe semantics. NaN/Inf/signed-zero exactness is not promised without explicit tests.
```

### 9.4 `vc4kernel.fragment_cmp`

```mlir
%p = vc4kernel.fragment_cmp %a, %b {predicate = #vc4kernel.cmp<slt>} : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
%q = vc4kernel.fragment_cmp %x, %y {predicate = #vc4kernel.cmp<olt>, fp_policy = #vc4kernel.fp_cmp_policy<finite_only>} : vector<16xf32>, vector<16xf32> -> !vc4kernel.pred<16>
```

Supported integer predicates:

```text
eq
ne
ult
ule
ugt
uge
slt
sle
sgt
sge
```

Supported f32 predicates:

```text
oeq
one
olt
ole
ogt
oge
```

Rules:

```text
- Integer predicates require vector<16xi32> operands and must not carry fp_policy.
- f32 predicates require vector<16xf32> operands and fp_policy = finite_only in P3.
- f32 support is finite-only ordered comparison. NaN-sensitive and unordered forms deterministically reject until a later explicit policy phase.
- Result is normally class general_mask unless the planner proves a more structured class.
```

### 9.5 `vc4kernel.fragment_select`

```mlir
%r = vc4kernel.fragment_select %p, %true, %false
  : !vc4kernel.pred<16>, vector<16xT>, vector<16xT> -> vector<16xT>
```

Allowed `T`:

```text
i32
f32
```

Semantics:

```text
lane l = true[l] if p[l] is active, else false[l]
```

Accepts all predicate classes.

### 9.6 `vc4kernel.fragment_rotate`

```mlir
%r = vc4kernel.fragment_rotate %value {amount = 1 : i32}
  : vector<16xT> -> vector<16xT>
```

Allowed `T`:

```text
i32
f32
```

Rules:

```text
- amount must be an integer attribute in [0, 15].
- Rotation direction must be documented and proven by lit plus hardware tests.
- No dynamic rotate amount is permitted in this stage.
```

### 9.7 `vc4kernel.fragment_reduce`

```mlir
%r = vc4kernel.fragment_reduce %value, %pred {kind = #vc4kernel.reduce<add>}
  : vector<16xT>, !vc4kernel.pred<16> -> vector<16xT>
```

Allowed `T`:

```text
i32
f32
```

Semantics:

```text
Computes the sum of active lanes and returns a vector<16xT> in which every lane contains the same reduced value.
Inactive lanes do not contribute.
If pred is empty, the result is zero for add.
```

Lowering uses a fragment_select-to-zero plus rotate/ALU tree. There is no first-class VC4 hardware reduction instruction in v1.

---

## 10. Address and offset conventions

### 10.1 Raw device pointer convention

All global memory base pointers inside `vc4kernel` are scalar `i32` raw VC4 device pointer words. This matches `vc4_deviceptr_t` in the runtime. No `memref` values exist inside `vc4kernel`.

### 10.2 Byte offsets

All offsets consumed by `vc4kernel` global memory ops are byte offsets.

Rules:

```text
- No memory op accepts element offsets.
- No memory op has offset_unit="element".
- Byte offsets must be 4-byte aligned for 32-bit executable semantics.
- Element-indexed semantics belong above vc4kernel in the memref/vector value layer.
```

### 10.3 Fragment byte-offset values

Per-lane byte offsets are represented as:

```text
vector<16xi32>
```

They are data fragments, not vector-dialect computations.

### 10.4 Contiguity classes

```text
general per-lane offsets:
  byte_offsets[l] can vary arbitrarily but must be aligned for active lanes.

contiguous 32-bit row fragment:
  byte_offsets[l] = base_byte_offset + 4*l for all lanes in the row.

row-tail fragment:
  contiguous row fragment with full/empty/tail_prefix predicate.
```

Rules:

```text
- TMU load may accept general per-lane byte offsets.
- VDW direct DMA store paths require contiguous row/rectangle memory. vdw_store_fragment v1 supports full/tail/rect preserve semantics; arbitrary sparse masks deterministic-reject until a later hardware-proven phase.
- VDR global-to-VPM supports full rectangular memory movement only.
- Arbitrary scatter stores are not part of v1.
```

---

## 11. VPM/VDR/VDW hardware mode model

VPM/VDR/VDW setup semantics are defined by the VideoCore IV Architecture
Reference Guide, Section 7 / Tables 31-37. That architecture reference is
normative for hardware field meanings; dialect docs describe the compiler
contract and must not invent alternate field semantics.

### 11.1 Hardware-derived lower-half schema

SSAVC4 and scheduled VC4 must model VPM/VDR/VDW using hardware-derived attrs/enums rather than old string compatibility names.

Required enums/attrs at lower-half level, with corresponding vc4kernel attrs where exposed:

```text
orientation:
  horizontal
  vertical

width:
  w32
  w16
  w8

subword_mode:
  none
  packed
  laned

coordinate fields:
  x
  y

VPM QPU read/write fields:
  orientation
  width
  subword_mode
  x
  y
  stride
  num_vectors, for grouped setup where applicable

VDR fields:
  width
  orientation
  x
  y
  row_len
  nrows
  memory_pitch_bytes
  vpm_pitch
  extended_memory_stride_bytes, if needed

VDW fields:
  width
  orientation
  x
  y
  units_or_nrows
  depth_or_row_len
  memory_stride_bytes
  block_mode
```

For VDW, the hardware STRIDE field is the 13-bit byte gap from the last byte
of one stored row to the first byte of the next stored row. High-level row pitch
must be translated through row_bytes/gap semantics before encoding. No
0xffff/65535 VDW stride model is allowed.

### 11.2 Executable v1 precision policy for modes

The schema must include `w8`, `w16`, `packed`, and `laned` to avoid an artificial lower-half shape, but executable `vc4kernel` v1 may use only:

```text
width = w32
subword_mode = none
orientation = horizontal | vertical
stride / pitch / blockmode where meaningful for 32-bit movement
```

Sub-32 packed/laned executable movement is a future precision/storage milestone. In 32-bit mode, the hardware ignores the laned bit, so executable 32-bit ops must require `subword_mode = none`.

### 11.3 VPM coordinate rules for 32-bit mode

For QPU VPM read/write:

```text
horizontal 32-bit:
  x must be 0 for a full 16-lane row fragment.
  y selects the VPM row.

vertical 32-bit:
  x selects the column.
  y must be aligned as required by the hardware vertical 32-bit address encoding.
```

All row/coordinate accesses are relative to the runtime-assigned `vpm_base_row` plus planned allocation offsets.

Full 64-row VPM capacity is available to a block when the resource plan allows
it. Per-operation executable v1 shape is still bounded to at most one 16x16
w32/none rectangle, so larger VPM use is expressed as multiple hardware-shaped
tiles, for example row bases 0, 16, 32, and 48.

---

## 12. Memory and VPM operations

### 12.1 Naming principle

Operation names must encode VC4 path/planning intent, not generic producer intent.

Use explicit path names:

```text
tmu_load_fragment
vdw_store_fragment
vpm_alloc
vpm_write_fragment
vpm_read_fragment
vdr_load_to_vpm
vdw_store_vpm
vdw_store_vpm_fragment
```

Do not add producer-facing names such as `masked_load_global`, `masked_store_global`, `tile_load`, `tile_store`, or `copy_tile`.

### 12.2 `vc4kernel.tmu_load_fragment`

```mlir
%value = vc4kernel.tmu_load_fragment %base, %byte_offsets, %pred
  : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xT>
```

Allowed `T`:

```text
i32
f32
```

Semantics:

```text
global memory -> register fragment through TMU direct-address load.
For each active lane l: load 32 bits from base + byte_offsets[l].
For each inactive lane: result lane is zero for the element type.
```

Rules:

```text
- base is raw i32 device pointer word.
- byte_offsets are byte offsets.
- active byte offsets must be 4-byte aligned.
- all predicate classes are legal.
- empty predicates must not issue a TMU request.
- tail/general masks must use explicit safe inactive-address policy after P7 before zero-selecting inactive lanes. Until P7, existing implicit safe-address behavior is only a migration target and must not be expanded.
```

### 12.3 `vc4kernel.vdw_store_fragment`

```mlir
vc4kernel.vdw_store_fragment %base, %byte_offsets, %value, %pred
  : i32, vector<16xi32>, vector<16xT>, !vc4kernel.pred<16>
```

Allowed `T`:

```text
i32
f32
```

Semantics:

```text
Planning op for register fragment -> global memory through compiler-managed VPM staging and VDW.
For each active lane l: store value[l] to base + byte_offsets[l].
For each inactive lane: preserve destination memory.
```

Rules:

```text
- byte_offsets must describe a contiguous 32-bit row fragment.
- pred may be full, empty, tail_prefix, or rect_row in executable v1.
- full/tail may lower to direct staging plus active-prefix store if hardware path preserves inactive destination.
- arbitrary sparse general_mask must deterministic-reject until the deferred sparse VDW phase exists.
- empty predicate skips the store.
- This op must never lower as if VC4 had a direct register-to-global store instruction.
```

### 12.4 `vc4kernel.vpm_alloc`

```mlir
%tile = vc4kernel.vpm_alloc {rows = 16 : i32, elem_bytes = 4 : i32}
  : !vc4kernel.vpm_tile
```

Semantics:

```text
Declares a statically planned VPM allocation in the current logical request/block.
```

Rules:

```text
- rows is an integer attribute in [1, 64].
- elem_bytes must be exactly 4 in executable v1.
- The op is a resource declaration, not a runtime heap allocation.
- The lowering assigns non-overlapping row offsets relative to vpm_base_row.
- Explicit user rows are accounted separately from compiler hidden staging rows.
```

### 12.5 `vc4kernel.vpm_write_fragment`

```mlir
vc4kernel.vpm_write_fragment %tile, %y, %x, %value, %pred
  {orientation = #vc4kernel.vpm_orientation<horizontal>,
   width = #vc4kernel.vpm_width<w32>,
   subword_mode = #vc4kernel.vpm_subword<none>}
  : !vc4kernel.vpm_tile, i32, i32, vector<16xT>, !vc4kernel.pred<16>
```

Allowed `T`:

```text
i32
f32
```

Semantics:

```text
register fragment -> VPM.
Active lanes write value[l].
Inactive lanes write zero.
```

Rules:

```text
- y/x identify a location inside the planned VPM allocation, relative to that allocation.
- orientation horizontal or vertical is legal in v1.
- width must be w32 in executable v1.
- subword_mode must be none in executable v1.
- all predicate classes are legal.
```

### 12.6 `vc4kernel.vpm_read_fragment`

```mlir
%value = vc4kernel.vpm_read_fragment %tile, %y, %x, %pred
  {orientation = #vc4kernel.vpm_orientation<horizontal>,
   width = #vc4kernel.vpm_width<w32>,
   subword_mode = #vc4kernel.vpm_subword<none>}
  : !vc4kernel.vpm_tile, i32, i32, !vc4kernel.pred<16> -> vector<16xT>
```

Allowed `T`:

```text
i32
f32
```

Semantics:

```text
VPM -> register fragment.
Active lanes read VPM data.
Inactive lanes return zero.
```

Rules match `vpm_write_fragment` for coordinate/mode legality.

### 12.7 `vc4kernel.vdr_load_to_vpm`

```mlir
vc4kernel.vdr_load_to_vpm %base, %byte_offset, %tile, %dst_y, %dst_x
  {rows = 4 : i32,
   cols = 16 : i32,
   memory_pitch_bytes = 64 : i32,
   vpm_pitch = 16 : i32,
   orientation = #vc4kernel.vpm_orientation<horizontal>,
   width = #vc4kernel.vpm_width<w32>,
   subword_mode = #vc4kernel.vpm_subword<none>}
  : i32, i32, !vc4kernel.vpm_tile, i32, i32
```

Semantics:

```text
global memory -> VPM through VDR/VCD DMA.
Copies a full static 32-bit rectangular block from global memory into VPM.
```

Rules:

```text
- base is raw i32 device pointer word.
- byte_offset is scalar byte offset and must be 4-byte aligned.
- rows > 0.
- cols in [1, 16] for one row fragment width; future extensions may expose larger rectangles by multiple rows.
- memory_pitch_bytes is a positive 4-byte-aligned integer attr.
- vpm_pitch is a hardware VPM pitch value; v1 supports 32-bit legal pitch modes.
- dst_y/dst_x plus rectangle dimensions must fit the allocation.
- no predicate operand is accepted.
- runtime rectangular/tail loads use `vc4kernel.vdr_load_rect_to_vpm`; the static op remains for full fixed-size rectangles.
```

### 12.8 `vc4kernel.vdw_store_vpm`

```mlir
vc4kernel.vdw_store_vpm %tile, %src_y, %src_x, %base, %byte_offset
  {rows = 4 : i32,
   cols = 16 : i32,
   memory_stride_bytes = 0 : i32,
   block_mode = false,
   orientation = #vc4kernel.vpm_orientation<horizontal>,
   width = #vc4kernel.vpm_width<w32>,
   subword_mode = #vc4kernel.vpm_subword<none>}
  : !vc4kernel.vpm_tile, i32, i32, i32, i32
```

Semantics:

```text
VPM rectangle -> global memory through VDW DMA.
Full rectangular store. No general lane predicate.
```

Rules:

```text
- source coordinates plus rectangle dimensions must fit the allocation.
- base is raw i32 device pointer word.
- byte_offset is scalar byte offset and must be 4-byte aligned.
- rows/cols describe the full DMA rectangle.
- width must be w32 in executable v1.
- subword_mode must be none in executable v1.
```

### 12.9 `vc4kernel.vdw_store_vpm_fragment`

```mlir
vc4kernel.vdw_store_vpm_fragment %tile, %src_y, %src_x, %base, %byte_offset, %pred
  {orientation = #vc4kernel.vpm_orientation<horizontal>,
   width = #vc4kernel.vpm_width<w32>,
   subword_mode = #vc4kernel.vpm_subword<none>}
  : !vc4kernel.vpm_tile, i32, i32, i32, i32, !vc4kernel.pred<16>
```

Semantics:

```text
One VPM fragment -> global memory through VDW.
Active lanes store VPM words.
Inactive lanes preserve destination memory.
```

Rules:

```text
- full/empty/tail_prefix are direct classes.
- arbitrary sparse general_mask deterministic-rejects until the deferred sparse VDW phase is implemented and hardware-proven.
- byte_offset must be 4-byte aligned.
```

This op is for shared/VPM-to-global fragment paths. Register-to-global convenience paths use `vdw_store_fragment` and compiler-managed staging.

---

## Dynamic rectangular global<->VPM transfer planning

VC4Kernel supports two classes of VDR/VDW movement.

### Static rectangular movement

`vc4kernel.vdr_load_to_vpm` and `vc4kernel.vdw_store_vpm_fragment` continue to represent fully static, hardware-shaped 32-bit rectangular movement. Their row/column shape and memory pitch are known as attributes. They are useful for fixed-size fixtures, static/full interior tiles, and fully specialized kernels. Static exact rectangular paths and dynamic rectangular paths both remain part of the final design.

### Dynamic rectangular movement

Blocked kernels such as GEMV and GEMM require runtime problem sizes and runtime leading dimensions. The correct VC4Kernel representation is not padded input matrices and not TMU-to-register-to-VPM as the primary shared-memory path. The correct representation is a dynamic rectangular transfer plan.

The architecture-backed dynamic rectangular contract is:

1. Runtime problem sizes and leading dimensions are supported below VC4Kernel via dynamic rectangular transfer primitives; tile sizes remain compile-time.
2. VDR global-to-VPM loads support runtime `active_rows`, runtime `active_cols`, runtime `memory_pitch_bytes`, and device-side zero-fill for inactive or out-of-bounds VPM cells.
3. VDW VPM-to-global stores support runtime `active_rows`, runtime `active_cols`, runtime `memory_stride_bytes`, dynamic/nonzero VPM source row where supported, and preserve-destination semantics outside active rows/columns.
4. VDW STRIDE is the hardware 13-bit gap from the last byte of one row to the start of the next row. `memory_stride_bytes` is a high-level row pitch and must be translated to row_bytes/gap semantics; no 0xffff/65535 VDW stride model is allowed.
5. VPM/VDR/VDW setup semantics are defined by the VideoCore IV Architecture Reference Guide, Section 7 / Tables 31-37. The architecture reference is normative for hardware field meanings.
6. Full 64-row VPM capacity must be usable through multiple 16x16 tiles, such as rows 0, 16, 32, and 48; per-op max shape remains <=16x16 w32/none for v1.
7. Row-by-row fallback is allowed only for true hardware-unencodable overflow cases, and each fallback class must be explicit and hardware-tested.
8. Static exact rectangular paths and dynamic rectangular paths both remain part of the final design. Static/full interior tiles keep compile-time information; dynamic/tail/runtime tiles use dynamic rect ops.
9. Blocked GEMV/GEMM must use natural VDR global-to-VPM shared-memory loads, not TMU-to-VPM as a workaround. TMU may still be used for unrelated direct register loads; it is not an accepted sparse VDW preserve workaround before the deferred sparse-store phase.
10. Planned-emission branch-count invariant: dynamic VDR/VDW branch-critical regions must be self-counted by planned emission or documented as non-branch-critical.

The op semantics are independent of GEMM/GEMV; GEMM/GEMV are only the first workloads that force the general feature. Active rows and columns are runtime scalar `i32` values. Memory pitch/stride in bytes is a runtime scalar `i32` value.

### `vc4kernel.vdr_load_rect_to_vpm`

```mlir
vc4kernel.vdr_load_rect_to_vpm
  %base, %byte_offset, %tile, %dst_row,
  %active_rows, %active_cols, %memory_pitch_bytes
  {
    max_rows = 16 : i32,
    max_cols = 16 : i32,
    elem_bytes = 4 : i32,
    orientation = #vc4kernel.vpm_orientation<horizontal>,
    width = #vc4kernel.vpm_width<w32>,
    subword = #vc4kernel.vpm_subword<none>,
    dst_x = 0 : i32,
    vpm_pitch = 1 : i32
  }
  : i32, i32, !vc4kernel.vpm_tile, i32, i32, i32, i32
```

Semantics:

For `0 <= r < max_rows` and `0 <= c < max_cols`, if `r < clamp(active_rows,0,max_rows)` and `c < clamp(active_cols,0,max_cols)`, load the 32-bit element at:

```text
base + byte_offset + r * memory_pitch_bytes + c * 4
```

into the destination VPM rectangle. Otherwise write zero into that destination VPM element. `memory_pitch_bytes` must be 4-byte aligned at runtime.

### `vc4kernel.vdw_store_rect_from_vpm`

```mlir
vc4kernel.vdw_store_rect_from_vpm
  %tile, %src_row, %base, %byte_offset,
  %active_rows, %active_cols, %memory_stride_bytes
  {
    max_rows = 16 : i32,
    max_cols = 16 : i32,
    elem_bytes = 4 : i32,
    orientation = #vc4kernel.vpm_orientation<horizontal>,
    width = #vc4kernel.vpm_width<w32>,
    subword = #vc4kernel.vpm_subword<none>,
    src_x = 0 : i32,
    vpm_pitch = 1 : i32
  }
  : !vc4kernel.vpm_tile, i32, i32, i32, i32, i32, i32
```

Semantics:

For active rows/columns, store VPM elements to global memory at:

```text
base + byte_offset + r * memory_stride_bytes + c * 4
```

For inactive rows/columns, preserve destination memory. `memory_stride_bytes` must be 4-byte aligned at runtime.

Verifier requirements:

- `max_rows` and `max_cols` are positive attrs and each is <= 16 for v1.
- `elem_bytes = 4`, `width = w32`, `subword = none`.
- `dst_row/src_row` must be a scalar i32 constant within the VPM allocation for v1.
- `dst_x/src_x` must be in hardware-valid range.
- `vpm_pitch` must be positive.
- `active_rows`, `active_cols`, and pitch/stride operands are scalar i32.
- Static memory-pitch attrs remain legal only on static ops.
- Dynamic rect ops are the preferred form for runtime problem sizes and leading dimensions.

---

## 13. Synchronization

### 13.1 `vc4kernel.barrier`

```mlir
vc4kernel.barrier
```

Semantics:

```text
Cooperative-block barrier equivalent to VC4-safe __syncthreads-like synchronization for all resident warps in the current logical block.
```

Rules:

```text
- Legal only in schedule_mode=cooperative_block kernels.
- The compiler computes uses_barrier=true below vc4kernel.
- The compiler computes semaphore_count_per_block=4 for the v1 barrier protocol.
- The runtime assigns semaphore_base per resident cooperative block.
- Raw semaphore protocol remains below vc4kernel.
```

Barrier lowering must use `warp_id`, `warps_per_block`, and `semaphore_base` through SSAVC4/lower-half mechanisms.

---

## 14. Control flow

### 14.1 Raw SCF forbidden

No raw `scf.*` op is legal inside verified `vc4kernel` IR. Structured control flow must be lowered before entering `vc4kernel` or by an explicit pre-verification helper pass.

### 14.2 Legal terminators

```text
vc4kernel.return
cf.br
cf.cond_br
```

### 14.3 Branch operands

Allowed successor operand/block-arg types:

```text
i1
i32
f32
vector<16xi32>
vector<16xf32>
!vc4kernel.pred<16>
```

`!vc4kernel.vpm_tile` block arguments should be rejected in v1 unless the verifier proves they are the same static allocation handle. The conservative locked behavior is to reject them.

### 14.4 Branch conditions

`cf.cond_br` condition must be scalar `i1`.

Predicates must be converted to scalar conditions through:

```mlir
%i1 = vc4kernel.pred.any %p : !vc4kernel.pred<16> -> i1
%i1 = vc4kernel.pred.all %p : !vc4kernel.pred<16> -> i1
```

Scalar `i1` lowering uses condition plans and hardware flags/branch conditions.

---

## 15. Attributes and enums

### 15.1 Required enum attributes

```text
#vc4kernel.schedule_mode<independent_vector | cooperative_block>
#vc4kernel.reduce<add>
#vc4kernel.cmp<eq | ne | ult | ule | ugt | uge>
#vc4kernel.vpm_orientation<horizontal | vertical>
#vc4kernel.vpm_width<w32 | w16 | w8>
#vc4kernel.vpm_subword<none | packed | laned>
```

Executable v1 accepts only:

```text
vpm_width = w32
vpm_subword = none
```

### 15.2 Forbidden old surface attrs

Final `vc4kernel` must not define or depend on old ergonomic-surface attributes:

```text
layout
role
precision
packing
storage_type as a generic tile attr
expressed_type as a generic tile attr
accumulator_type as a generic tile attr
boundary_policy as generic tile policy
tile shape descriptors
rank descriptors
strides descriptors as generic layout algebra
```

Concrete hardware mode information belongs in concrete memory path ops or lower-half VPM/VDR/VDW mode attrs.

### 15.3 Precision policy

Executable `vc4kernel` semantics are 32-bit only:

```text
i32 / u32 represented as i32
f32
vector<16xi32>
vector<16xf32>
32-bit global/VPM/VDR/VDW movement
```

Forbidden executable precision:

```text
f16
bf16
fp8
fp4
i8/u8
i16/u16
int4/uint4
packed/nibble movement
quantization scale/zero-point lowering
```

Sub-32 hardware mode attrs may exist in the lower half but must be rejected for executable vc4kernel v1.

---

## 16. Operation inventory summary

### 16.1 Kernel and identity

```text
vc4kernel.kernel
vc4kernel.return
vc4kernel.program_id
vc4kernel.num_programs
vc4kernel.warp_id
vc4kernel.lane_range
```

Explicitly removed:

```text
block-id operation
lane-id operation
vc4kernel.thread_id
```

### 16.2 Predicates

```text
vc4kernel.pred.full
vc4kernel.pred.empty
vc4kernel.pred.tail
vc4kernel.pred.rect
vc4kernel.pred.and
vc4kernel.pred.or
vc4kernel.pred.not
vc4kernel.pred.any
vc4kernel.pred.all
```

### 16.3 Fragment values and compute

```text
vc4kernel.splat
vc4kernel.fragment_alu.add
vc4kernel.fragment_alu.mul
vc4kernel.fragment_cmp
vc4kernel.fragment_select
vc4kernel.fragment_rotate
vc4kernel.fragment_reduce
```

### 16.4 Memory/resources

```text
vc4kernel.tmu_load_fragment
vc4kernel.vdw_store_fragment
vc4kernel.vpm_alloc
vc4kernel.vpm_write_fragment
vc4kernel.vpm_read_fragment
vc4kernel.vdr_load_to_vpm
vc4kernel.vdr_load_rect_to_vpm
vc4kernel.vdw_store_vpm
vc4kernel.vdw_store_vpm_fragment
vc4kernel.vdw_store_rect_from_vpm
```

### 16.5 Synchronization

```text
vc4kernel.barrier
```

### 16.6 Explicitly absent from final inventory

```text
vc4kernel.thread_id
block-id operation
lane-id operation
vc4kernel.tile_load
vc4kernel.tile_store
vc4kernel.copy_tile
vc4kernel.tile_view
vc4kernel.tile_subview
vc4kernel.transpose_view
vc4kernel.tile_fill
vc4kernel.tile_broadcast
vc4kernel.tile_add
vc4kernel.tile_sub
vc4kernel.tile_mul
vc4kernel.tile_select
vc4kernel.tile_reduce
vc4kernel.row_reduce
vc4kernel.warp_reduce
vc4kernel.block_reduce
vc4kernel.tile_dot
vc4kernel.tile_contract
vc4kernel.tile_matmul
vc4kernel.surface_placeholder
vc4kernel.memref_load
vc4kernel.memref_store
vc4kernel.vector_transfer_read
vc4kernel.vector_transfer_write
vc4kernel.fragment_contract
```

`fragment_contract` is forbidden as a permanent Surface v2 `vc4kernel` feature. Future vector contract handling must lower through the standard value layer into admitted target execution-plan features instead of resurrecting a contract-specific VC4Kernel op.

---

## 17. Verifier contract

`--verify-vc4kernel` must enforce all of the following.

### 17.1 Module boundary

```text
- module top-level operations must be vc4kernel.kernel only.
- every non-kernel vc4kernel op must be nested inside exactly one vc4kernel.kernel.
- nested vc4kernel.kernel ops are forbidden.
```

### 17.2 Dialect boundary

Reject any operation inside `vc4kernel.kernel` not explicitly allowed by this document.

Required diagnostic classes:

```text
raw scf forbidden
vector dialect forbidden
memref/tensor forbidden
producer dialect forbidden
ssavc4/vc4 forbidden
unknown vc4kernel op forbidden
old tile surface op forbidden
thread_id/block_id/lane_id forbidden
sub-32 type forbidden
index type forbidden
```

### 17.3 Type legality

Reject:

```text
index-typed values anywhere
vector<16xi1>
vector width != 16
vector element not i32/f32
memref/tensor/pointer types
sub-32 scalar/vector types
unknown vc4kernel types
```

### 17.4 Kernel metadata

Verify:

```text
- public_name exists and is a non-empty string.
- schedule_mode is present and valid.
- warps_per_block is present and valid.
- independent_vector has warps_per_block=1.
- cooperative_block has 1 <= warps_per_block <= 12.
- arg_attrs length equals entry block argument count.
- each arg_attrs dictionary has required fields.
- buffer args are i32 formal args.
- scalar args have matching i32/f32 formal types.
- user-authored resource dictionaries are forbidden in source vc4kernel.
```

### 17.5 Predicate legality

Verify:

```text
- every pred op produces !vc4kernel.pred<16>.
- vector<16xi1> masks are rejected.
- branching on pred directly is rejected.
- consumer-specific predicate legality is enforced.
- general_mask is legal only where the consumer has specified support. Arbitrary sparse VDW store consumers deterministic-reject until the deferred sparse VDW phase.
```

### 17.6 Memory legality

Verify:

```text
- global bases are i32.
- global offsets are byte offsets.
- active byte offsets are 4-byte aligned, or the path must reject if alignment cannot be proven.
- vdw_store_fragment offsets are contiguous row fragments.
- arbitrary scatter global stores are forbidden.
- vdr_load_to_vpm is full rectangular and statically shaped.
- vpm coordinates are within allocation bounds where statically provable; dynamic/unproven coordinates must be rejected unless a range proof exists.
- VPM allocations plus required hidden staging must fit the 64-row v1 planning window.
- inactive global store semantics preserve destination.
- inactive load and VPM read semantics zero-fill.
- inactive VPM write semantics write zero.
```

### 17.7 Control-flow legality

```text
- only allowed terminators are vc4kernel.return, cf.br, cf.cond_br.
- cf.cond_br condition is scalar i1.
- successor operands and block args use legal types.
- no unresolved raw scf remains.
- no illegal region nesting or non-isolated capture exists.
- VPM handle block args are rejected in v1.
```

### 17.8 Lowering-boundary legality

Reject any `vc4kernel` IR that would require:

```text
- direct vc4kernel -> scheduled vc4 lowering
- host-side computation
- fixture-name dispatch
- unknown hardware path
- unimplemented VDR/VDW/TMU/VPM mode
- executable sub-32 precision
- arbitrary scatter global stores
- dynamic VPM layout not proven by hardware/range analysis
```

---

## 18. `vc4kernel -> ssavc4` lowering contract

### 18.1 Required input

The conversion pass accepts verified `vc4kernel` IR only and must reject malformed IR with deterministic diagnostics.

### 18.2 Required output

Successful conversion emits:

```text
ssavc4.module
ssavc4.func
ssavc4 operations
vc4.launch_abi metadata on ssavc4.func
computed vc4.resource metadata on ssavc4.func
```

No `vc4kernel.*` op may remain.

### 18.3 Launch ABI lowering

```text
formal args -> vc4.launch_abi.args[] -> sequential ssavc4.uniform.read
program_id  -> builtin uniform when used
warp_id     -> builtin uniform when cooperative/local-warp identity is used
vpm_base_row -> builtin uniform when total_vpm_rows_per_block > 0
semaphore_base -> builtin uniform when semaphore_count_per_block > 0
lane_range  -> ssavc4.element_number-derived vector
```

Scalar `f32` formals lower to scalar `f32` uniform reads. Scalar `i1` values lower through condition plans, not as ordinary data unless materialization is explicitly needed.

Physical `QPU_NUMBER` must not be used for logical identity or canonical VPM ownership. VPM ownership is via runtime-assigned `vpm_base_row`.

### 18.4 Resource lowering

The pass computes the semantic `vc4.resource` summary described in §6.5 and emits it on the SSAVC4 function. Scheduled VC4/codegen/runtime must consume this semantic summary.

### 18.5 Fragment lowering

Representative mappings:

```text
vc4kernel.splat            -> ssavc4.splat / scalar uniform splat / load immediate as appropriate
fragment_alu.add           -> ssavc4.alu.add with mapped VC4 add-pipe opcode
fragment_alu.mul           -> ssavc4.alu.mul with mapped VC4 mul-pipe opcode
explicit i32 mul32         -> tested add/and/shr/shl plus mul24 decomposition
vc4kernel.fragment_cmp     -> predicate plan / flags / mask value as needed
vc4kernel.fragment_select  -> conditional select using predicate plan
vc4kernel.fragment_rotate  -> ssavc4.rotate
vc4kernel.fragment_reduce  -> zero-mask then rotate/ALU tree
```

### 18.6 Predicate lowering

Predicates lower to structured active-lane counts, mask values, flags, branch conditions, or select/merge logic depending on the consumer. No `!vc4kernel.pred<16>` value may survive into SSAVC4.

Required semantic preservation:

```text
TMU inactive lanes -> zero result and safe/no inactive memory requests
VPM read inactive lanes -> zero
VPM write inactive lanes -> zero written to VPM
VDW/global store inactive lanes -> preserve destination memory
pred.any/pred.all -> correct scalar i1 via any/all hardware flag logic
```

### 18.7 Memory lowering

```text
tmu_load_fragment:
  TMU request/read with explicit safe inactive-address behavior and zero-fill after P7; current implicit behavior is a migration target.

vdw_store_fragment:
  compiler-managed hidden VPM staging; full/tail/rect preserve semantics are v1, while arbitrary sparse masks deterministic-reject until the deferred sparse VDW phase.

vpm_alloc:
  allocation table/resource planning only; no runtime heap op.

vpm_write_fragment / vpm_read_fragment:
  SSAVC4 VPM setup/read/write using hardware mode attrs and vpm_base_row-relative coordinates.

vdr_load_to_vpm:
  SSAVC4 VDR/VCD global-to-VPM setup/address/wait sequence using hardware mode attrs.

vdw_store_vpm / vdw_store_vpm_fragment:
  SSAVC4 VDW setup/address/wait sequence using hardware mode attrs; preserve inactive destination where predicated.

barrier:
  SSAVC4 barrier/semaphore sequence using semaphore_base, warp_id, and warps_per_block.
```

### 18.8 CFG lowering

`cf.br` and `cf.cond_br` lower to SSAVC4 branch ops with successor operands/block args. If scheduled-VC4 constraints require block layout or branch-delay-slot transformations, they belong in SSAVC4-to-VC4/scheduling machinery or a deterministic conversion subpass, not in fixture-specific hacks.

### 18.9 No shortcuts

Forbidden conversion behavior:

```text
- scheduled vc4 ops in vc4kernel -> ssavc4 output
- direct QASM emission
- fixture-name special cases
- public_name special cases
- expected JSON edits
- reference bundle substitution
- host-side computation to satisfy tests
- weakened scheduled VC4 verifier rules
```

---

## 19. Relationship to future standard vector/memref layer

The future value layer handles:

```text
memref<?xi32, #vc4.global>
memref<?xf32, #vc4.global>
vector.transfer_read
vector.transfer_write
vector.contract
vector.reduction
vector.mask
vector.transpose
vector.step
arith on vector types
scf structured control flow
```

Vector-to-vc4kernel lowering responsibilities:

```text
memref raw pointer arg      -> i32 formal arg with buffer arg_attrs
vector.step                 -> vc4kernel.lane_range
vector<16xi1> masks         -> !vc4kernel.pred<16>
vector.transfer_read        -> tmu_load_fragment or VDR+VPM path
vector.transfer_write       -> vdw_store_fragment or explicit VPM+VDW path
vector arithmetic           -> fragment ops
vector reduction            -> fragment_reduce
scf                         -> cf before verified vc4kernel
```

Triton lowering responsibilities after vector is locked:

```text
tt.get_program_id(axis=0..2) -> program_id {axis = 0|1|2}
tt.num_programs(axis=0..2)  -> num_programs {axis = 0|1|2}
tt.make_range / arange      -> vector.step -> lane_range
tt.load mask/other          -> vector.transfer_read -> vc4kernel predicate + TMU/VDR plan
tt.store mask               -> vector.transfer_write -> vc4kernel predicate + VDW plan
```

Do not lower TTIR directly into old tile surface ops or directly into SSAVC4/scheduled VC4 as an accepted path.

---

## 20. Required tests and acceptance

### 20.1 Dialect roundtrip tests

Required source tests include:

```text
kernel-roundtrip.mlir
identity-roundtrip.mlir
predicate-roundtrip.mlir
fragment-ops-roundtrip.mlir
tmu-vdw-roundtrip.mlir
vpm-roundtrip.mlir
vdr-vpm-roundtrip.mlir
barrier-roundtrip.mlir
resource-computation-roundtrip-or-diagnostic.mlir
```

Tests must be updated to remove `block_id`, `lane_id`, and source-authored resource dictionaries.

### 20.2 Invalid diagnostics

Required invalid tests include:

```text
invalid-forbidden-vector-op.mlir
invalid-forbidden-memref.mlir
invalid-forbidden-scf.mlir
invalid-forbidden-thread-id.mlir
invalid-forbidden-block-id.mlir
invalid-forbidden-lane-id.mlir
invalid-index-type.mlir
invalid-vector-width.mlir
invalid-vector-i1-mask.mlir
invalid-sub32-type.mlir
invalid-kernel-arg-attrs.mlir
invalid-user-resource-metadata.mlir
invalid-vdw-noncontiguous-store.mlir
invalid-vdr-predicate.mlir
invalid-vpm-out-of-bounds.mlir
invalid-barrier-independent-vector.mlir
invalid-top-level-nonkernel-op.mlir
invalid-stray-vc4kernel-op-outside-kernel.mlir
invalid-builtin-unrealized-cast.mlir
invalid-arith-cmpi-non-i32.mlir
invalid-vpm-dynamic-coordinate-unproven.mlir
invalid-vpm-handle-block-arg.mlir
invalid-lower-half-dialect-ssavc4-vc4.mlir
invalid-producer-dialect-tt-gpu-linalg.mlir
invalid-vdr-vdw-vpm-unaligned-offset.mlir
invalid-vdw-unaligned-or-unknown-base.mlir
invalid-subword-executable-vpm-mode.mlir
```

`invalid-predicate-nonnormalizable` from the old spec should be replaced or revised: non-normalizable/general masks are now legal where consumers have a fallback, and illegal only for consumers that cannot support them.

### 20.3 Conversion tests

Required conversion lit files include:

```text
minimal-kernel.mlir
formal-args-launch-abi.mlir
identity-lowering.mlir
lane-range-lowering.mlir
predicate-tail-lowering.mlir
predicate-general-mask-lowering.mlir
fragment-cmp-select-lowering.mlir
fragment-arith-lowering.mlir
fragment-imul32-lowering.mlir
fragment-rotate-lowering.mlir
fragment-reduce-lowering.mlir
tmu-load-fragment-full-tail-general.mlir
vdw-store-fragment-staging.mlir
vdw-store-fragment-sparse-mask-reject.mlir
vpm-read-write-horizontal.mlir
vpm-read-write-vertical.mlir
vdr-load-to-vpm-horizontal-vertical.mlir
vdw-store-vpm-horizontal-vertical.mlir
barrier-lowering.mlir
control-flow-block-args.mlir
reject-forbidden-surface-ops.mlir
reject-direct-scheduled-vc4.mlir
```

### 20.4 Hardware fixtures

Every executable feature requires real hardware proof.

Minimum hardware fixtures:

```text
vc4kernel_vector_store_full
vc4kernel_vector_store_tail_preserve
vc4kernel_vector_store_rect_preserve
vc4kernel_vector_store_sparse_mask_reject
vc4kernel_program_id_writeback
vc4kernel_tmu_load_full
vc4kernel_tmu_load_tail_zero_fill
vc4kernel_tmu_load_general_mask_zero_fill
vc4kernel_saxpy_f32
vc4kernel_f32_scalar_uniform_splat
vc4kernel_imul32_fast_mul24
vc4kernel_imul32_software_fallback
vc4kernel_fragment_rotate
vc4kernel_fragment_reduce_sum
vc4kernel_vpm_roundtrip_horizontal_32
vc4kernel_vpm_roundtrip_vertical_32
vc4kernel_vdr_to_vpm_horizontal_32
vc4kernel_vdr_to_vpm_vertical_32
vc4kernel_vpm_to_global_vdw_horizontal_32
vc4kernel_vpm_to_global_vdw_vertical_32
vc4kernel_control_flow_block_args
vc4kernel_qpu_barrier_syncthreads
```

Every hardware fixture must:

```text
- generate fresh candidate artifacts from vc4kernel input
- lower through vc4kernel -> ssavc4 -> scheduled vc4
- assemble/build/run on real hardware
- compare copied-back device output to a CPU reference
- check sentinel preservation for predicated stores
- check zero-fill behavior for predicated loads
- not use reference QASM
- not fake VC4_TEST_RESULT
- not compute the output on the host harness instead of the device
```

### 20.5 Static anti-shortcut tests

The suite must scan for and reject:

```text
- direct vc4kernel -> scheduled VC4 conversion path
- producer lowering before the vector/Triton stages
- executable sub-32 lowering
- old tile surface op definitions
- vc4kernel.thread_id / block_id / lane_id
- vector dialect ops allowed by verifier
- memref/tensor/scf/tt/gpu inside verified vc4kernel
- fixture-name/public_name special casing
- comments or dummy literals added solely to satisfy scans
- stale candidate reuse in acceptance gates
```

### 20.6 Final Stage 1 acceptance definition

`vc4kernel` Stage 1 is accepted only when all of these are true:

```text
1. The dialect exists as vc4kernel with final names and namespace.
2. The ODS op/type/attr inventory matches this document.
3. FunctionOpInterface and function_type behavior are coherent.
4. The verifier rejects every forbidden dialect/type/op class listed here.
5. No producer-facing surface op exists in vc4kernel.
6. No thread-id, block-id, or lane-id operation exists.
7. No vector dialect op is legal in vc4kernel.
8. No memref/tensor/scf/tt/gpu/linalg/producers are legal in vc4kernel.
9. Raw vector<16xi1> masks are replaced by !vc4kernel.pred<16>.
10. General predicates and fragment_cmp are supported where consumers define fallbacks.
11. Global memory ops use explicit TMU/VDR/VDW path names.
12. Register->global stores lower through compiler-managed VPM staging.
13. VPM ops use !vc4kernel.vpm_tile and hardware-derived mode attrs.
14. Resource metadata is computed, not source-authored.
15. Runtime-assigned vpm_base_row and semaphore_base are supported where required.
16. Conversion is vc4kernel -> ssavc4 only.
17. No direct vc4kernel -> scheduled vc4 path exists.
18. Required lit tests pass.
19. Required hardware fixtures pass on real VC4 hardware.
20. check-vc4 remains green.
21. Anti-shortcut/integrity scans pass.
```

---

## 21. Future revisions outside this spec

Outside the current executable spec:

```text
- Triton TTIR parsing/registration/lowering implementation
- IREE/Linalg lowering
- standard vector/memref/arith -> vc4kernel implementation
- executable f16/bf16/fp8/int8/int4/sub-32 precision
- quantization
- packed/laned executable VPM movement
- arbitrary global scatter stores
- arbitrary sparse VDW stores before the deferred sparse-store phase
- integer div/mod unless a library sequence is designed
- atomics
- tile-buffer color/Z/stencil
- texture filtering / cube maps / varyings for compute v1
- fragment_contract / matmul-specific VC4Kernel op
```

These may be added only by separate design documents and vertical hardware-proven implementation slices, except forbidden permanent Surface v2 features such as `fragment_contract`, `tile_dot`, `tile_matmul`, and `tile_contract`, which must not be resurrected as VC4Kernel ops.

---

## 22. Final one-paragraph contract

`vc4kernel` is a strict VC4 target-kernel planning dialect below standard MLIR vector/memref/arith/scf/cf and above SSAVC4. It contains only a function-like kernel wrapper, formal ABI metadata, logical program/warp identity, SIMD lane ranges, 16-lane data fragments, 16-lane predicate/mask values, explicit TMU/VDR/VPM/VDW memory path operations, VPM allocation handles, compiler-managed VPM staging requirements, barriers, scalar CFG, and target fragment arithmetic/reduction operations. It contains no old tile surface, no memref, no vector dialect operations, no raw SCF, no Triton/IREE/producers, no thread_id/block_id/lane_id, no vector<16xi1> masks, and no executable sub-32 precision. It computes resource requirements instead of accepting source-authored resource summaries. Its only valid lower path is `vc4kernel -> ssavc4 -> scheduled vc4 -> artifacts/runtime/hardware`.

---

## References used for this replacement spec

- Current old strict spec attached in this conversation: `vc4kernel_dialect_strict_specification.md`, 2026-05-31 upload.
- Earlier strict spec and rescope roadmap attached in this conversation.
- Broadcom, *VideoCore IV 3D Architecture Reference Guide*, especially QPU SIMD/uniform/condition/branch sections and VPM/VDR/VDW setup tables.
- vc4asm documentation: <https://www.maazl.de/project/vc4asm/doc/index.html> and <https://www.maazl.de/project/vc4asm/doc/vc4.qinc.html>.
- Triton documentation for `tt.load`, `tt.store`, `tt.make_range`, `triton.Config`, TritonGPU local-memory/memdesc ops, and NVIDIA backend metadata extraction/packing.
