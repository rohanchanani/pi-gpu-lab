# VC4Kernel Dialect Strict Specification

**Status:** locked Stage 1 target specification for the `vc4kernel` dialect.  
**Date:** 2026-05-30.  
**Audience:** implementation agents that will create `vc4kernel` by copying `vc4tile`, then pruning, renaming, adapting, and adding code until the result exactly matches this document.  
**Primary purpose:** define the final `vc4kernel` dialect contract, independent of compatibility with existing `vc4tile` names or M5 surface choices.

---

## 0. Normative rule

This document is the source of truth for `vc4kernel`.

The implementation must not preserve an existing `vc4tile` operation, attribute, type, verifier rule, pass, or lowering behavior merely because it exists. Existing `vc4tile` is only implementation source material. The final `vc4kernel` dialect must match this specification.

Anything not explicitly permitted by this document is forbidden in verified `vc4kernel` IR.

The required final stack is:

```text
Triton-emitted TTIR / future IREE-Linalg-value IR
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
1. Specify vc4kernel exactly.                         <-- this document
2. Create vc4kernel by copying vc4tile.
3. Prune / rename / adapt vc4kernel until it matches this spec.
4. Prove vc4kernel -> ssavc4 -> scheduled vc4 -> hardware.
5. Only then add standard vector/memref/arith -> vc4kernel.
6. Only then add real Triton-emitted TTIR -> standard value layer.
```

---

## 1. Core identity of the dialect

### 1.1 What `vc4kernel` is

`vc4kernel` is the VC4 target-kernel planning dialect immediately above SSAVC4.

It represents a kernel after generic value-layer semantics have been lowered into VC4-specific execution fragments, memory paths, predicates, launch identity, VPM resources, and synchronization/resource requirements.

It answers:

> How should this already-tiled, already-vectorized kernel execute on VC4 QPUs, VPM, TMU, VDR/VCD, VDW, uniforms, and semaphores?

### 1.2 What `vc4kernel` is not

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
```

### 1.3 Required boundary separation

The three major compiler layers must stay distinct:

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
  VC4 predicates
  raw device-pointer words
  TMU/VDR/VPM/VDW path choice
  VPM allocation and access
  program/block/warp/lane identity
  launch/resource metadata
  barriers
  target fragment arithmetic/reductions
  core CFG

ssavc4:
  target machine SSA
  uniforms
  element_number
  ALU dataflow
  flags and branches
  TMU requests/reads
  VPM reads/writes
  VDR loads
  VDW stores
  semaphores/barriers
  thread_end
```

No `vc4kernel` pass may lower directly to scheduled `vc4`. The only valid lower path is:

```text
vc4kernel -> ssavc4 -> scheduled vc4 -> artifacts/runtime/hardware
```

---

## 2. Source material and transformation strategy

The implementation should create `vc4kernel` by copying the current `vc4tile` dialect and conversion structure, then editing the copy until it exactly matches this document.

The copy-from-`vc4tile` strategy is an implementation-risk reduction technique only. It does not constrain this specification.

### 2.1 Source material to preserve semantically

Current `vc4tile` contains useful target-kernel material that should be reworked into `vc4kernel`:

```text
kernel wrapper / return
formal kernel ABI args
program/block/warp/lane identity
lane range
mask and tail-mask concepts
global load/store core paths
rotate/reduce core operations
shared/VPM allocation and access
VDR/VCD and VDW experience
barrier and cooperative resources
core CFG / block-argument lowering
vc4tile -> ssavc4 conversion structure
hardware candidate runner discipline
```

### 2.2 Source material that must not survive as `vc4kernel`

The old ergonomic `vc4tile` surface must not survive in the final `vc4kernel` dialect:

```text
tile_descriptor
tile_load
tile_store
copy_tile
tile_view
tile_subview
transpose_view
shared_tile_alloc as ergonomic tile allocation
tile_fill
tile_broadcast
tile_add
tile_sub
tile_mul
tile_select
tile_reduce as ergonomic tile surface
row_reduce as ergonomic tile surface
warp_reduce as ergonomic tile surface
block_reduce as ergonomic tile surface
tile_dot
tile_contract
tile_matmul
surface_placeholder
layout algebra attributes
role metadata attributes
precision/packing metadata fields copied from M5 surface
```

These concepts belong either above `vc4kernel` in standard MLIR (`vector`, `memref`, `arith`, `scf/cf`) or later as planned low-level `vc4kernel` fragment/resource operations with different semantics.

---

## 3. Names, files, and pass names

### 3.1 Dialect identity

Final dialect identity:

```text
MLIR operation prefix: vc4kernel.
MLIR type prefix:      !vc4kernel.*
MLIR attribute prefix: #vc4kernel.*
C++ namespace:         mlir::vc4kernel
TableGen prefix:       VC4Kernel
Dialect class:         VC4KernelDialect
```

### 3.2 Expected file layout

The final implementation should use this file layout:

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

Final required passes:

```text
--verify-vc4kernel
--convert-vc4kernel-to-ssavc4
```

Optional but allowed helper passes:

```text
--legalize-vc4kernel-cfg
```

`--legalize-vc4kernel-cfg` is allowed only if it lowers a small remaining CFG legality detail inside `vc4kernel`. It must not accept raw `scf.*` in verified core. Structured `scf` lowering belongs above `vc4kernel`.

These pass names must not be aliases to old `vc4tile` passes in the final implementation. Temporary aliases may be used only during early migration, but the completed Stage 1 implementation must own real `vc4kernel` pass registrations and tests.

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
i1   scalar branch/condition/result of scalar predicates
i32  signed or unsigned 32-bit integer / raw device pointer word / byte offset / index / resource count
f32  32-bit floating-point scalar
```

`u32` is represented as MLIR `i32` with unsigned interpretation documented by operation semantics or ABI metadata. There is no distinct unsigned MLIR builtin type.

### 4.2 Legal fragment carrier types

Only these vector carrier types are legal:

```text
vector<16xi32>
vector<16xf32>
```

These are data carriers for one VC4 QPU 16-lane fragment. They are legal types even though `vector.*` dialect operations are forbidden inside verified `vc4kernel` IR.

### 4.3 Legal `vc4kernel` types

Final custom types:

```text
!vc4kernel.pred<16>
!vc4kernel.vpm_tile
```

No other `vc4kernel` types are legal unless this document is revised.

### 4.4 Predicate type

`!vc4kernel.pred<16>` is the only legal predicate type.

It is not equivalent to `vector<16xi1>`.

It represents a VC4-planned predicate over the current 16-lane fragment. It carries enough semantic information for the verifier/planner/lowering to distinguish legal VC4 predicate classes such as:

```text
full fragment
empty fragment
contiguous tail fragment
rectangular row fragment projected onto current lanes
normalizable Boolean composition of supported classes
```

It must be rejected if it cannot be normalized into a class accepted by a consuming operation.

### 4.5 VPM handle type

`!vc4kernel.vpm_tile` is an opaque handle to a statically planned VPM allocation inside a kernel/block.

It is a VC4 resource handle. It is not:

```text
- a tensor type
- a memref type
- a CuTe layout object
- a generic tile descriptor
- an owning heap allocation
```

The allocation op determines rows, element width, and scope. The type itself is intentionally simple.

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
VC4 scheduled dialect types
```

Sub-32 precision is rejected before or at the boundary into `vc4kernel`. There are no executable sub-32 types in this dialect.

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

### 5.2 Allowed `arith` operations

Only scalar `arith` operations are allowed:

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
- arith.cmpi may compare only scalar i32 values and produce scalar i1.
- arith.select may select only scalar i1/i32/f32 values.
```

All vector/fragment arithmetic must use `vc4kernel.fragment_*` operations.

### 5.3 Allowed `cf` operations

Allowed CFG operations:

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
```

### 5.4 Forbidden dialects

The following dialects are always forbidden inside verified `vc4kernel` IR:

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

The dialect verifier must reject them with deterministic diagnostics.

Important: `vector<16xi32>` and `vector<16xf32>` types are legal; `vector.*` operations are not.

---

## 6. Kernel operation and launch/resource metadata

### 6.1 `vc4kernel.kernel`

Summary:

```text
Top-level VC4 target-kernel planning operation.
```

Required traits/interfaces:

```text
IsolatedFromAbove
Symbol
Function-like region with no results
```

Final syntax target:

```mlir
vc4kernel.kernel @name(%arg0: i32, %arg1: f32) attributes {
  public_name = "name",
  schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
  arg_attrs = [ ... ],
  resource = { ... }
} {
^entry(%arg0: i32, %arg1: f32):
  ...
  vc4kernel.return
}
```

The exact printer may use standard function-like assembly, but the semantics must match this contract.

### 6.2 Kernel formal arguments

Formal arguments are user/caller-provided values. They lower to `vc4.launch_abi.args[]` on the generated SSAVC4 function.

Legal formal argument types:

```text
i32
f32
```

`i32` may represent:

```text
- by-value integer scalar
- raw `vc4_deviceptr_t` device pointer word
- by-value unsigned scalar
```

`f32` represents a by-value 32-bit float scalar.

No formal argument may be a memref, tensor, vector, predicate, VPM handle, index, or sub-32 type.

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
- No full host-side memref descriptor is ever represented in vc4kernel.
- uniform_index is not user-authored in vc4kernel source. It is assigned during vc4kernel -> ssavc4 launch ABI construction.
- The verifier must reject missing, malformed, duplicate, or length-mismatched arg_attrs.
```

### 6.4 Builtin identity values

The following ops are zero-operand runtime identity ops and lower to launch builtins or lane machinery:

```text
vc4kernel.program_id
vc4kernel.block_id
vc4kernel.warp_id
vc4kernel.lane_id
vc4kernel.lane_range
```

They are not formal kernel arguments. They must not carry `uniform_index`. They must not be interpreted as arbitrary user values.

Lowering categories:

```text
program_id -> vc4.launch_abi.builtins logical_request-like uniform
block_id   -> vc4.launch_abi.builtins logical_block_id-like uniform
warp_id    -> vc4.launch_abi.builtins logical_warp_id-like uniform
lane_id    -> derived from ssavc4.element_number, not a uniform
lane_range -> derived from ssavc4.element_number / lane vector construction, not a uniform
```

### 6.5 `thread_id` is forbidden

There is no `vc4kernel.thread_id` op.

If a thread-like scalar is needed, compute it explicitly from legal identity values:

```text
warp_id * 16 + lane_id
program_id * block_size + lane_range
```

The dialect must reject any copied `vc4tile.thread_id` operation if it survives the port.

Rationale:

```text
Triton exposes program IDs and vector/block offsets.
The standard value layer exposes vector-shaped lanes.
VC4 hardware exposes QPU lane identity.
A CUDA thread_id primitive is not fundamental here and risks reintroducing a wrong SIMT abstraction.
```

### 6.6 Schedule mode

`vc4kernel.kernel` must carry:

```text
schedule_mode = #vc4kernel.schedule_mode<independent_vector>
```

or:

```text
schedule_mode = #vc4kernel.schedule_mode<cooperative_block>
```

Semantics:

```text
independent_vector:
  each logical request is independently schedulable over the 12 QPU warp slots.
  barriers are forbidden.
  VPM use is permitted only if the verifier proves no cross-request shared-state hazard or the resource policy reserves isolated rows.

cooperative_block:
  a block consists of 1..12 logical warps that must be resident together if barriers or shared VPM communication are used.
  block_id and warp_id are meaningful.
  barriers and shared VPM require full-block residency.
```

### 6.7 Resource metadata

`vc4kernel.kernel` must carry a resource dictionary or equivalent first-class attrs with these fields:

```text
warps_per_block_max:           i32, required, 1..12
uses_vpm:                      bool, required
uses_barrier:                  bool, required
require_full_block_residency:  bool, required
vpm_rows_per_block:            i32, required, 0..64
vpm_bytes_per_block:           i32, required, must equal vpm_rows_per_block * 16 * 4 unless 0
semaphores_per_block:          i32, required, 0 or 4 for barrier-using kernels unless future spec revises
```

Rules:

```text
- If uses_barrier=true, schedule_mode must be cooperative_block.
- If uses_barrier=true, require_full_block_residency must be true.
- If uses_barrier=true, semaphores_per_block must be 4.
- If uses_vpm=true, vpm_rows_per_block must be >0.
- If vpm_rows_per_block >0, uses_vpm must be true.
- vpm_rows_per_block must not exceed 64.
- vpm_bytes_per_block must not exceed 4096.
- warps_per_block_max must be 1 for independent_vector unless a later verified independent multi-warp policy is added.
- warps_per_block_max must be 1..12 for cooperative_block.
```

### 6.8 `vc4kernel.return`

Terminator for `vc4kernel.kernel` blocks that exit the kernel.

Rules:

```text
- Takes no operands.
- Returns no values.
- Must appear only inside vc4kernel.kernel.
- Lowers to ssavc4.thread_end on terminal paths.
```

---

## 7. Identity and lane operations

### 7.1 `vc4kernel.program_id`

Signature:

```mlir
%pid = vc4kernel.program_id : i32
```

Semantics:

```text
Logical program/request index for the current kernel launch.
For Triton axis-0 v1 lowering, tt.get_program_id(axis=0) maps here.
It is not physical QPU_NUMBER.
It is not a formal argument.
```

### 7.2 `vc4kernel.block_id`

Signature:

```mlir
%bid = vc4kernel.block_id : i32
```

Semantics:

```text
Logical cooperative block id.
Valid primarily for cooperative_block schedule mode.
May be used in independent_vector only if the verifier/lowering has a documented mapping.
```

### 7.3 `vc4kernel.warp_id`

Signature:

```mlir
%wid = vc4kernel.warp_id : i32
```

Semantics:

```text
Logical warp/QPU slot id within a cooperative block.
Range is 0 <= warp_id < warps_per_block.
Not physical QPU_NUMBER.
```

### 7.4 `vc4kernel.lane_id`

Signature:

```mlir
%lane = vc4kernel.lane_id : i32
```

Semantics:

```text
Current scalar lane id, 0..15, derived from QPU lane element number.
Not a uniform.
```

### 7.5 `vc4kernel.lane_range`

Signature:

```mlir
%lanes = vc4kernel.lane_range : vector<16xi32>
```

Semantics:

```text
Vector [0, 1, 2, ..., 15] in lane order.
Used to form fragment offsets and lane-wise values.
```

---

## 8. Predicate model

### 8.1 Predicate design goal

`vc4kernel` predicates are target-planned predicates, not generic vector masks.

They must preserve enough structure to let the verifier and lowering decide whether a memory path is legal on VC4:

```text
TMU direct per-lane load
VDW contiguous store
VDR full-rectangle load to VPM
VPM row/column read/write
barrier/skipping control flow
fragment select/reduce
```

### 8.2 Predicate classes

Every `!vc4kernel.pred<16>` value must be normalizable into one of these classes:

```text
full:
  all 16 lanes active

empty:
  no lanes active

tail:
  active lanes form a contiguous prefix [0, width), where 0 <= width <= 16

rect_row:
  predicate for one row of a rectangular tile; in a single 16-lane fragment this is equivalent to full, empty, or tail but carries row/bounds provenance for diagnostics

normal_composition:
  Boolean composition of full/empty/tail/rect_row that normalizes to full, empty, or tail
```

Sparse arbitrary masks are not part of the initial final `vc4kernel` contract. If a future milestone supports sparse fallback, it must add an explicit op/class such as `vc4kernel.pred.sparse` and a corresponding lowering contract. Until then, non-normalizable predicates are rejected.

### 8.3 `vc4kernel.pred.full`

Signature:

```mlir
%p = vc4kernel.pred.full : !vc4kernel.pred<16>
```

Semantics: all lanes active.

### 8.4 `vc4kernel.pred.empty`

Signature:

```mlir
%p = vc4kernel.pred.empty : !vc4kernel.pred<16>
```

Semantics: no lanes active.

### 8.5 `vc4kernel.pred.tail`

Signature:

```mlir
%p = vc4kernel.pred.tail %base, %limit : i32, i32 -> !vc4kernel.pred<16>
```

Semantics:

```text
lane l is active iff %base + l < %limit, with unsigned/nonnegative semantics established by verifier constraints.
```

Rules:

```text
- %base and %limit are scalar i32.
- The operation represents a contiguous prefix over lane_range.
- If the verifier can prove base >= limit, the predicate may canonicalize to empty.
- If the verifier can prove base + 15 < limit, the predicate may canonicalize to full.
```

### 8.6 `vc4kernel.pred.rect`

Signature:

```mlir
%p = vc4kernel.pred.rect %row, %rows, %col_base, %cols
  : i32, i32, i32, i32 -> !vc4kernel.pred<16>
```

Semantics:

```text
lane l is active iff:
  %row < %rows and %col_base + l < %cols
```

This is the canonical predicate for one 16-lane row-fragment of a rectangular/tail tile.

Rules:

```text
- row/rows/col_base/cols are scalar i32.
- The result must normalize to empty, full, or tail for the current fragment.
- This op does not represent an entire 2D mask at once; it represents the predicate for the current 16-lane fragment/row.
```

### 8.7 Boolean predicate operations

Allowed operations:

```mlir
%p = vc4kernel.pred.and %a, %b : !vc4kernel.pred<16>
%p = vc4kernel.pred.or  %a, %b : !vc4kernel.pred<16>
%p = vc4kernel.pred.not %a     : !vc4kernel.pred<16>
```

Rules:

```text
- Results must remain normalizable into an accepted predicate class.
- Non-normalizable results are rejected by --verify-vc4kernel or by the first consumer that requires a stricter class.
- Boolean ops must canonicalize constants: full/empty identities, double-not, associative flattening.
```

### 8.8 Predicate-to-scalar operations

Allowed operations:

```mlir
%b = vc4kernel.pred.any %p : !vc4kernel.pred<16> -> i1
%b = vc4kernel.pred.all %p : !vc4kernel.pred<16> -> i1
```

Semantics:

```text
pred.any is true if at least one lane is active.
pred.all is true if every lane is active.
```

These are the only way to branch on a predicate in `vc4kernel`.

### 8.9 Forbidden predicate representations

Forbidden:

```text
vector<16xi1> as a mask type
arith.cmpi producing vector<16xi1>
vector.mask
vector.create_mask
raw integer bitmasks
opaque predicate attributes on memory ops instead of !vc4kernel.pred<16> values
```

---

## 9. Fragment value operations

### 9.1 Fragment operation rule

All operations over `vector<16xi32>` or `vector<16xf32>` values inside `vc4kernel` must be `vc4kernel` operations.

No `arith.*` op may operate on vector types.

### 9.2 `vc4kernel.splat`

Signature:

```mlir
%v = vc4kernel.splat %x : i32 -> vector<16xi32>
%v = vc4kernel.splat %x : f32 -> vector<16xf32>
```

Semantics:

```text
Broadcast scalar value to all 16 lanes.
```

This replaces all uses of `vector.broadcast` or `vector.splat` inside `vc4kernel`.

### 9.3 Arithmetic fragment ops

Allowed operations:

```mlir
%r = vc4kernel.fragment_add %a, %b : vector<16xi32>, vector<16xi32> -> vector<16xi32>
%r = vc4kernel.fragment_add %a, %b : vector<16xf32>, vector<16xf32> -> vector<16xf32>
%r = vc4kernel.fragment_sub %a, %b : vector<16xi32>, vector<16xi32> -> vector<16xi32>
%r = vc4kernel.fragment_sub %a, %b : vector<16xf32>, vector<16xf32> -> vector<16xf32>
%r = vc4kernel.fragment_mul %a, %b : vector<16xi32>, vector<16xi32> -> vector<16xi32>
%r = vc4kernel.fragment_mul %a, %b : vector<16xf32>, vector<16xf32> -> vector<16xf32>
%r = vc4kernel.fragment_shl %a, %amount : vector<16xi32>, i32 -> vector<16xi32>
```

Rules:

```text
- Operand and result element types must match.
- i32 addition/subtraction/multiplication use 32-bit modular integer semantics unless a consumer/verifier documents unsigned interpretation.
- f32 addition/subtraction/multiplication use VC4 f32 semantics exposed by the lower half.
- fragment_shl is i32 only. Shift amount must be a scalar i32 constant in the initial implementation unless later verified otherwise.
```

### 9.4 `vc4kernel.fragment_cmp`

Signature:

```mlir
%p = vc4kernel.fragment_cmp %a, %b {predicate = #vc4kernel.cmp<eq>}
  : vector<16xi32>, vector<16xi32> -> !vc4kernel.pred<16>
```

Supported integer predicates:

```text
eq
ne
ult
ule
ugt
uge
```

Rules:

```text
- Initial support is i32 fragments only.
- f32 fragment compare is forbidden until a later spec revision.
- Result must be normalizable or legal for the consuming operation.
```

### 9.5 `vc4kernel.fragment_select`

Signature:

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
lane l = true[l] if p[l] active, else false[l]
```

### 9.6 `vc4kernel.fragment_rotate`

Signature:

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
- Rotation direction must be documented and consistent with existing SSAVC4 rotate. The final implementation must choose one direction and prove it with tests.
- No dynamic rotate amount is permitted in the initial final spec.
```

### 9.7 `vc4kernel.fragment_reduce`

Signature:

```mlir
%r = vc4kernel.fragment_reduce %value, %pred {kind = #vc4kernel.reduce<add>}
  : vector<16xT>, !vc4kernel.pred<16> -> vector<16xT>
```

Allowed `T`:

```text
i32
f32
```

Supported reduction kinds in this spec:

```text
add
```

Semantics:

```text
Computes the sum of active lanes and returns a vector<16xT> in which every lane contains the same reduced value.
Inactive lanes do not contribute to the sum.
If pred is empty, the result is zero for add.
```

Rationale:

```text
Returning a broadcast fragment keeps the operation implementable on VC4 SIMD lanes and lets later code choose which lane(s) to store using predicates.
```

Future scalar-returning reduction forms are not part of this spec.

---

## 10. Address and offset conventions

### 10.1 Raw device pointer convention

All global memory base pointers inside `vc4kernel` are scalar `i32` raw VC4 device pointer words.

This matches `vc4_deviceptr_t` in the runtime.

No `memref` values exist inside `vc4kernel`.

### 10.2 Byte offsets

All offsets consumed by `vc4kernel` global memory ops are byte offsets.

Rules:

```text
- No memory op accepts element offsets.
- No memory op has offset_unit="element".
- Byte offsets must be 4-byte aligned in current 32-bit executable semantics.
- Element-indexed semantics belong above vc4kernel in the memref/vector value layer.
```

### 10.3 Fragment byte-offset values

Per-lane byte offsets are represented as:

```text
vector<16xi32>
```

They are data fragments, not vector-dialect computations. They must be produced through `vc4kernel` identity/splat/fragment arithmetic or through future vector-to-vc4kernel lowering.

### 10.4 Contiguity requirements

Some memory paths require contiguous row fragments. The verifier must enforce this.

Definitions:

```text
contiguous 32-bit row fragment:
  byte_offsets[l] = base_byte_offset + 4*l for active lanes

row-tail fragment:
  contiguous row fragment with predicate full/empty/tail
```

Rules:

```text
- TMU load may accept general per-lane byte offsets if the lower half supports the pattern.
- VDW store must reject non-contiguous byte offsets.
- VDR global-to-VPM must reject non-rectangular/non-full fragments.
- VPM row/column access must reject predicates or orientations that would require unsupported dynamic VPM layout.
```

---

## 11. Global/register/VPM memory operations

### 11.1 Memory operation naming principle

Final `vc4kernel` memory op names must encode the VC4 path, not generic intent.

Use:

```text
tmu_load_fragment
vdw_store_fragment
vdr_load_to_vpm
vpm_alloc
vpm_read_fragment
vpm_write_fragment
vdw_store_vpm_fragment
```

Do not use ambiguous final names such as:

```text
masked_load_global
masked_store_global
global_load_fragment
global_store_fragment
copy_tile
tile_load
tile_store
```

Those names may be implementation source material only.

### 11.2 `vc4kernel.tmu_load_fragment`

Signature:

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
global memory -> register fragment through the TMU/direct load path.
For each active lane l:
  load 32 bits from address base + byte_offsets[l]
For each inactive lane:
  result lane is zero for i32 or +0.0 for f32.
```

Rules:

```text
- base is a raw i32 device pointer word.
- byte_offsets is vector<16xi32> in bytes.
- byte_offsets must be 4-byte aligned for all active lanes.
- pred must be normalizable.
- Result type selects interpretation of loaded raw32 bits.
- The op has memory read effects.
```

Lowering obligation:

```text
Lower to ssavc4.tmu.request / ssavc4.tmu.read or equivalent SSAVC4 TMU path.
```

### 11.3 `vc4kernel.vdw_store_fragment`

Signature:

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
register fragment -> global memory through the VC4 VPM/VDW store path.
For each active lane l:
  store value[l] to address base + byte_offsets[l]
For each inactive lane:
  destination memory is preserved.
```

Rules:

```text
- byte_offsets must describe a contiguous 32-bit row fragment unless a later verified sparse/scatter store path is added.
- pred must normalize to full, empty, or tail for the current contiguous row.
- Non-contiguous stores are rejected.
- This op has memory write effects.
```

Lowering obligation:

```text
Lower through SSAVC4 VPM/VDW support. It must not pretend VC4 has a direct register-to-global store instruction.
```

### 11.4 `vc4kernel.vpm_alloc`

Signature:

```mlir
%tile = vc4kernel.vpm_alloc {rows = 16 : i32, elem_bytes = 4 : i32}
  : !vc4kernel.vpm_tile
```

Semantics:

```text
Declares a statically planned VPM allocation in the current kernel/block.
```

Rules:

```text
- rows must be an integer attribute in [1, 64].
- elem_bytes must be exactly 4 in this spec.
- Total bytes = rows * 16 * 4.
- The kernel resource metadata must account for this allocation.
- Multiple allocations are allowed only if their total rows do not exceed vpm_rows_per_block and the lowering assigns non-overlapping row ranges.
- The op is a resource declaration, not a runtime heap allocation.
```

### 11.5 `vc4kernel.vpm_write_fragment`

Signature:

```mlir
vc4kernel.vpm_write_fragment %tile, %row, %value, %pred
  {orientation = #vc4kernel.vpm_orientation<row>}
  : !vc4kernel.vpm_tile, i32, vector<16xT>, !vc4kernel.pred<16>
```

Allowed `T`:

```text
i32
f32
```

Supported orientations:

```text
row
column
```

Semantics:

```text
register fragment -> VPM.
Active lanes write value[l].
Inactive lanes write zero for the element type.
```

Rules:

```text
- row is scalar i32 and must be within the vpm_tile allocation.
- orientation=row writes one VPM row of up to 16 words.
- orientation=column writes one VPM column-style fragment only if supported by current SSAVC4/VC4 lowering; otherwise verifier rejects it.
- pred must be legal for the orientation.
- This op has VPM write effects.
```

Zero-fill for inactive lanes is intentional. If a future use case needs VPM preserve-on-inactive semantics, it must add an explicit new policy/op and prove it.

### 11.6 `vc4kernel.vpm_read_fragment`

Signature:

```mlir
%value = vc4kernel.vpm_read_fragment %tile, %row, %pred
  {orientation = #vc4kernel.vpm_orientation<row>}
  : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xT>
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
Inactive lanes return zero for the element type.
```

Rules:

```text
- row is scalar i32 and must be within the vpm_tile allocation.
- orientation=row reads one VPM row.
- orientation=column reads one VPM column-style fragment only if supported by current SSAVC4/VC4 lowering; otherwise verifier rejects it.
- pred must be legal for the orientation.
- This op has VPM read effects.
```

### 11.7 `vc4kernel.vdr_load_to_vpm`

Signature:

```mlir
vc4kernel.vdr_load_to_vpm %base, %byte_offset, %tile, %dst_row
  {rows = 4 : i32, cols = 16 : i32, global_stride_bytes = 64 : i32, elem_bytes = 4 : i32}
  : i32, i32, !vc4kernel.vpm_tile, i32
```

Semantics:

```text
global memory -> VPM through the VDR/VCD load path.
Copies a static row-major 32-bit rectangular block from global memory into consecutive VPM rows.
```

Rules:

```text
- base is a raw i32 device pointer word.
- byte_offset is scalar i32 in bytes.
- rows must be a positive integer attribute.
- cols must be an integer attribute in [1, 16].
- elem_bytes must be exactly 4.
- global_stride_bytes must be a positive 4-byte-aligned integer attribute.
- dst_row must be within the vpm_tile allocation, and dst_row + rows must not exceed allocation rows.
- This op supports only full rectangular VDR copies. Tail/partial copies must be decomposed before this op or handled by TMU + VPM write.
- No predicate operand is accepted.
- This op has global memory read and VPM write effects.
```

Lowering obligation:

```text
Lower to SSAVC4 VDR/VCD global-to-VPM support. If SSAVC4 lacks sufficient VDR support, Stage 1 must add it before this op is considered implemented.
```

### 11.8 `vc4kernel.vdw_store_vpm_fragment`

Signature:

```mlir
vc4kernel.vdw_store_vpm_fragment %tile, %src_row, %base, %byte_offset, %pred
  {elem_bytes = 4 : i32}
  : !vc4kernel.vpm_tile, i32, i32, i32, !vc4kernel.pred<16>
```

Semantics:

```text
VPM row fragment -> global memory through VDW.
Active lanes store words from VPM row to global memory.
Inactive lanes preserve destination memory.
```

Rules:

```text
- src_row must be within the vpm_tile allocation.
- base is a raw i32 device pointer word.
- byte_offset is scalar i32 in bytes.
- elem_bytes must be exactly 4.
- pred must normalize to full, empty, or tail.
- This op stores one contiguous row fragment only.
- This op has VPM read and global memory write effects.
```

This op is for shared/VPM-to-global paths. Register-to-global paths use `vdw_store_fragment`.

---

## 12. Synchronization

### 12.1 `vc4kernel.barrier`

Signature:

```mlir
vc4kernel.barrier
```

Semantics:

```text
Cooperative-block barrier equivalent to a VC4-safe __syncthreads-like synchronization for all resident warps in the current block.
```

Rules:

```text
- Legal only in schedule_mode=cooperative_block kernels.
- Kernel must have uses_barrier=true.
- Kernel must have require_full_block_residency=true.
- Kernel must reserve semaphores_per_block=4 unless future verifier proves another protocol.
- No barrier may appear in independent_vector kernels.
- The op has synchronization side effects.
```

Lowering obligation:

```text
Lower to ssavc4.barrier or equivalent SSAVC4 semaphore/barrier sequence.
The raw semaphore protocol must remain below vc4kernel.
```

---

## 13. Control flow

### 13.1 Raw SCF is forbidden

No raw `scf.*` op is legal inside verified `vc4kernel` IR.

Structured control flow must be lowered before entering `vc4kernel` or by an explicit pre-verification pass. Verified `vc4kernel` uses `cf` plus block arguments.

### 13.2 Legal terminators

Legal terminators in a `vc4kernel.kernel` region:

```text
vc4kernel.return
cf.br
cf.cond_br
```

### 13.3 Branch operands

`cf.br` and `cf.cond_br` may carry successor operands of legal `vc4kernel` types:

```text
i1
i32
f32
vector<16xi32>
vector<16xf32>
!vc4kernel.pred<16>
!vc4kernel.vpm_tile
```

However, passing `!vc4kernel.vpm_tile` through block arguments should be rejected unless the verifier proves it is the same static resource handle and not a dynamic runtime value.

### 13.4 Branch conditions

`cf.cond_br` condition must be scalar `i1`.

If a predicate controls branching, it must first use:

```mlir
%i1 = vc4kernel.pred.any %p : !vc4kernel.pred<16> -> i1
```

or:

```mlir
%i1 = vc4kernel.pred.all %p : !vc4kernel.pred<16> -> i1
```

### 13.5 Lowering obligation

`vc4kernel -> ssavc4` must preserve control-flow semantics through SSAVC4 successor operands/block arguments. It must not use fixture-name special cases or precomputed layout assumptions.

If SSAVC4 branch layout requires false-successor fallthrough or other scheduled constraints, the conversion must perform a deterministic block layout transformation or reject with a clear diagnostic. It must not silently generate invalid scheduled VC4.

---

## 14. Attributes and enums

### 14.1 Required enum attributes

The dialect must define at least these enums:

```text
#vc4kernel.schedule_mode<independent_vector | cooperative_block>
#vc4kernel.reduce<add>
#vc4kernel.cmp<eq | ne | ult | ule | ugt | uge>
#vc4kernel.vpm_orientation<row | column>
```

### 14.2 Forbidden old M5 surface attrs

Final `vc4kernel` must not define or depend on these old ergonomic-surface attributes:

```text
layout
role
precision
packing
storage_type as an attribute
expressed_type as an attribute
accumulator_type as an attribute
boundary_policy as generic tile policy
tile shape descriptors
rank descriptors
strides descriptors as generic layout algebra
```

If a path needs concrete stride or shape information, it must encode it directly in the relevant path op, such as:

```text
global_stride_bytes on vdr_load_to_vpm
rows/cols on vdr_load_to_vpm
rows on vpm_alloc
```

### 14.3 Precision policy

`vc4kernel` executable semantics are 32-bit only.

Allowed executable element/storage types:

```text
i32 / u32 represented as i32
f32
vector<16xi32>
vector<16xf32>
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
packed/nibble modes
quantization scale/zero-point lowering
```

No `vc4kernel` op may silently widen, narrow, pack, or unpack sub-32 types. Unsupported precision must be rejected before or at the `vc4kernel` verifier.

---

## 15. Operation inventory summary

This is the complete final `vc4kernel` op inventory for the initial strict dialect.

### 15.1 Kernel and identity

```text
vc4kernel.kernel
vc4kernel.return
vc4kernel.program_id
vc4kernel.block_id
vc4kernel.warp_id
vc4kernel.lane_id
vc4kernel.lane_range
```

### 15.2 Predicates

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

### 15.3 Fragment values and compute

```text
vc4kernel.splat
vc4kernel.fragment_add
vc4kernel.fragment_sub
vc4kernel.fragment_mul
vc4kernel.fragment_shl
vc4kernel.fragment_cmp
vc4kernel.fragment_select
vc4kernel.fragment_rotate
vc4kernel.fragment_reduce
```

### 15.4 Memory/resources

```text
vc4kernel.tmu_load_fragment
vc4kernel.vdw_store_fragment
vc4kernel.vpm_alloc
vc4kernel.vpm_write_fragment
vc4kernel.vpm_read_fragment
vc4kernel.vdr_load_to_vpm
vc4kernel.vdw_store_vpm_fragment
```

### 15.5 Synchronization

```text
vc4kernel.barrier
```

### 15.6 Explicitly absent from final inventory

The final initial dialect does not contain:

```text
vc4kernel.thread_id
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

`fragment_contract` is intentionally absent from the initial final spec. It may be added later only if the `vector.contract -> vc4kernel` lowering proves that a planned fragment-contract op is necessary and the op is specified separately.

---

## 16. Verifier contract

`--verify-vc4kernel` must enforce all of the following.

### 16.1 Dialect boundary

Reject any operation inside `vc4kernel.kernel` that is not explicitly allowed by this document.

Required diagnostic classes:

```text
raw scf forbidden
vector dialect forbidden
memref/tensor forbidden
producer dialect forbidden
ssavc4/vc4 forbidden
unknown vc4kernel op forbidden
old tile surface op forbidden
thread_id forbidden
sub-32 type forbidden
index type forbidden
```

### 16.2 Type legality

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

### 16.3 Kernel metadata

Verify:

```text
- public_name exists and is a non-empty string.
- schedule_mode is present and valid.
- arg_attrs length equals entry block argument count.
- each arg_attrs dictionary has required fields.
- buffer args are i32 formal args.
- scalar args have matching i32/f32 formal types.
- resource metadata is present and internally consistent.
- barrier/resource constraints are respected.
```

### 16.4 Predicate legality

Verify:

```text
- every pred op produces !vc4kernel.pred<16>.
- every pred consumer accepts the predicate's normalized class.
- non-normalizable Boolean predicate expressions are rejected.
- vector<16xi1> masks are rejected.
- branching on pred directly is rejected.
```

### 16.5 Memory legality

Verify:

```text
- global bases are i32.
- global offsets are byte offsets.
- byte offsets are 4-byte aligned when statically provable; otherwise lowering must preserve an alignment check or reject if the target path requires it.
- vdw_store_fragment offsets are contiguous row fragments.
- vdw_store_fragment predicates are full/empty/tail.
- vdr_load_to_vpm is full rectangular and statically shaped.
- vpm rows are within allocation bounds.
- VPM allocations fit kernel resources.
- inactive store semantics are preserve-global for global stores.
- inactive load semantics are zero-fill.
```

### 16.6 Control-flow legality

Verify:

```text
- only allowed terminators are vc4kernel.return, cf.br, cf.cond_br.
- cf.cond_br condition is scalar i1.
- successor operands and block args use legal types.
- no unresolved raw scf remains.
- no illegal region nesting or non-isolated capture exists.
```

### 16.7 Lowering-boundary legality

Reject any `vc4kernel` IR that would require:

```text
- direct vc4kernel -> scheduled vc4 lowering
- host-side computation
- fixture-name dispatch
- unknown hardware path
- unimplemented VDR/VDW/TMU/VPM mode
- executable sub-32 precision
- arbitrary sparse predication
- arbitrary scatter global stores
- dynamic VPM layout not proven by hardware
```

---

## 17. `vc4kernel -> ssavc4` lowering contract

### 17.1 Required input

The conversion pass accepts verified `vc4kernel` IR only.

It must reject unverified or malformed IR with deterministic diagnostics.

### 17.2 Required output

The conversion pass emits:

```text
ssavc4.module
ssavc4.func
ssavc4 operations
vc4.launch_abi metadata on ssavc4.func
vc4.resource metadata on ssavc4.func
```

No `vc4kernel.*` op may remain after successful conversion.

### 17.3 Launch ABI lowering

The pass must lower:

```text
vc4kernel formal args -> vc4.launch_abi.args[] -> ssavc4.uniform.read
vc4kernel.program_id  -> vc4.launch_abi.builtins[] -> ssavc4.uniform.read
vc4kernel.block_id    -> vc4.launch_abi.builtins[] -> ssavc4.uniform.read
vc4kernel.warp_id     -> vc4.launch_abi.builtins[] -> ssavc4.uniform.read
vc4kernel.lane_id     -> ssavc4.element_number-derived scalar value
vc4kernel.lane_range  -> ssavc4.element_number-derived vector value
```

The pass must not use physical QPU number for logical identity.

### 17.4 Resource lowering

The pass must lower resource metadata into the existing `vc4.resource` dictionary expected by SSAVC4 and scheduled VC4 emission.

Required mapping includes:

```text
schedule_mode
warps_per_block_max
uses_vpm / uses_shared_vpm equivalent
uses_barrier
require_full_block_residency
vpm_rows_per_block
vpm_bytes_per_block
semaphores_per_block
```

### 17.5 Fragment lowering

Representative mappings:

```text
vc4kernel.splat            -> ssavc4.splat or ssavc4.load_imm as appropriate
vc4kernel.fragment_add     -> ssavc4.alu.add
vc4kernel.fragment_sub     -> ssavc4.alu.add with subtract opcode/path if supported, otherwise deterministic reject
vc4kernel.fragment_mul     -> ssavc4.alu.mul
vc4kernel.fragment_shl     -> ssavc4.alu.add shl path or equivalent
vc4kernel.fragment_rotate  -> ssavc4.rotate
vc4kernel.fragment_reduce  -> rotate/ALU reduction sequence or existing ssavc4 reduction helper path
vc4kernel.fragment_select  -> flags/select sequence or deterministic reject if unsupported
```

### 17.6 Predicate lowering

Predicates lower to concrete SSAVC4 flags, active-lane metadata, branch conditions, or zero-fill/preserve logic depending on consumer.

No `!vc4kernel.pred<16>` value may survive into SSAVC4.

The lowering must preserve:

```text
load inactive lanes -> zero
VPM read inactive lanes -> zero
VPM write inactive lanes -> zero
VDW/global store inactive lanes -> preserve destination
```

### 17.7 Memory lowering

Representative mappings:

```text
vc4kernel.tmu_load_fragment       -> ssavc4.tmu.request + ssavc4.tmu.read
vc4kernel.vdw_store_fragment      -> VPM staging + ssavc4.vdw.store
vc4kernel.vpm_alloc               -> resource metadata / row assignment; no runtime heap op
vc4kernel.vpm_write_fragment      -> ssavc4.vpm.write
vc4kernel.vpm_read_fragment       -> ssavc4.vpm.read
vc4kernel.vdr_load_to_vpm         -> ssavc4 VDR/VCD global-to-VPM op/sequence
vc4kernel.vdw_store_vpm_fragment  -> ssavc4.vdw.store from planned VPM row
vc4kernel.barrier                 -> ssavc4.barrier
```

The conversion must fail rather than inventing unsupported hardware behavior.

### 17.8 CFG lowering

`cf.br` and `cf.cond_br` lower to SSAVC4 branch ops with successor operands/block args.

If SSAVC4/scheduled-VC4 requires a particular block layout, the conversion pass must produce that layout or reject.

### 17.9 No shortcuts

Forbidden conversion behavior:

```text
- no scheduled vc4 ops in vc4kernel -> ssavc4 output
- no direct QASM emission
- no fixture-name special cases
- no public_name special cases
- no expected JSON edits
- no reference bundle substitution
- no host-side computation to satisfy tests
- no weakening of scheduled VC4 verifier rules
```

---

## 18. Relationship to standard vector/memref layer

### 18.1 Future vector layer responsibilities

The future vector/memref/arith layer handles:

```text
memref<?xi32, #vc4.global>
memref<?xf32, #vc4.global>
vector.transfer_read
vector.transfer_write
vector.contract
vector.reduction
vector.mask
vector.transpose
arith on vector types
scf structured control flow
```

### 18.2 Vector-to-vc4kernel responsibilities

A later pass lowers standard value-layer operations to `vc4kernel`:

```text
memref raw pointer arg      -> i32 formal arg with buffer arg_attrs
vector.step/lane-like math  -> vc4kernel.lane_range and fragment ops
vector.transfer_read        -> vc4kernel.tmu_load_fragment or vdr_load_to_vpm + vpm_read
vector.transfer_write       -> vc4kernel.vdw_store_fragment or vdw_store_vpm_fragment
vector.reduction            -> vc4kernel.fragment_reduce
vector.contract             -> explicit vc4kernel fragment ops, or future fragment_contract if separately specified
vector.mask                 -> vc4kernel.pred.*
arith vector ops            -> vc4kernel.fragment_* ops
scf                         -> cf/block args before verified vc4kernel
```

### 18.3 What must not happen

Do not lower producers into old `tile_*` surface ops.

Do not put memref or vector operations in `vc4kernel` to avoid implementing the vector-to-vc4kernel pass.

Do not let `vc4kernel` become a second copy of the vector dialect.

---

## 19. Relationship to Triton TTIR

### 19.1 TTIR is not consumed by `vc4kernel` directly

Triton-emitted TTIR lowers first to the standard value layer:

```text
tt.get_program_id -> value-layer identity that later maps to vc4kernel.program_id
tt.arange         -> vector lane values / later vc4kernel.lane_range
tt.load           -> vector.transfer_read
tt.store          -> vector.transfer_write
tt.dot            -> vector.contract
```

Then value-layer lowering produces `vc4kernel`.

### 19.2 No TTIR/TTGIR in verified vc4kernel

`tt.*`, `ttg.*`, NVIDIA-specific TritonGPU, NVGPU, NVVM, ROCDL, or any other producer/target dialect is forbidden inside verified `vc4kernel` IR.

---

## 20. Examples

### 20.1 Minimal store kernel sketch

```mlir
module {
  vc4kernel.kernel @store16(%out : i32) attributes {
    public_name = "store16",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "out", kind = "buffer", direction = "out", elem_type = "u32"}],
    resource = {
      warps_per_block_max = 1 : i32,
      uses_vpm = false,
      uses_barrier = false,
      require_full_block_residency = false,
      vpm_rows_per_block = 0 : i32,
      vpm_bytes_per_block = 0 : i32,
      semaphores_per_block = 0 : i32
    }
  } {
  ^entry(%out : i32):
    %zero = arith.constant 0 : i32
    %two = arith.constant 2 : i32
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %byte_offsets = vc4kernel.fragment_shl %lanes, %two : vector<16xi32>, i32 -> vector<16xi32>
    %pred = vc4kernel.pred.full : !vc4kernel.pred<16>
    vc4kernel.vdw_store_fragment %out, %byte_offsets, %lanes, %pred
      : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
```

### 20.2 Tail load + store sketch

```mlir
%pid = vc4kernel.program_id : i32
%lanes = vc4kernel.lane_range : vector<16xi32>
%sixteen = arith.constant 16 : i32
%base_elem = arith.muli %pid, %sixteen : i32
%base_byte = arith.shli %base_elem, %two : i32
%base_byte_v = vc4kernel.splat %base_byte : i32 -> vector<16xi32>
%lane_bytes = vc4kernel.fragment_shl %lanes, %two : vector<16xi32>, i32 -> vector<16xi32>
%byte_offsets = vc4kernel.fragment_add %base_byte_v, %lane_bytes : vector<16xi32>, vector<16xi32> -> vector<16xi32>
%pred = vc4kernel.pred.tail %base_elem, %n : i32, i32 -> !vc4kernel.pred<16>
%xv = vc4kernel.tmu_load_fragment %x, %byte_offsets, %pred : i32, vector<16xi32>, !vc4kernel.pred<16> -> vector<16xf32>
vc4kernel.vdw_store_fragment %y, %byte_offsets, %xv, %pred : i32, vector<16xi32>, vector<16xf32>, !vc4kernel.pred<16>
```

### 20.3 VPM roundtrip sketch

```mlir
%tile = vc4kernel.vpm_alloc {rows = 1 : i32, elem_bytes = 4 : i32} : !vc4kernel.vpm_tile
%row0 = arith.constant 0 : i32
%pred = vc4kernel.pred.full : !vc4kernel.pred<16>
vc4kernel.vpm_write_fragment %tile, %row0, %value, %pred {orientation = #vc4kernel.vpm_orientation<row>}
  : !vc4kernel.vpm_tile, i32, vector<16xi32>, !vc4kernel.pred<16>
%read = vc4kernel.vpm_read_fragment %tile, %row0, %pred {orientation = #vc4kernel.vpm_orientation<row>}
  : !vc4kernel.vpm_tile, i32, !vc4kernel.pred<16> -> vector<16xi32>
```

---

## 21. Required tests for the dialect/conversion implementation

A `vc4kernel` implementation is not complete unless it includes tests covering these categories.

### 21.1 Dialect roundtrip tests

Required files under `compiler/test/Dialect/VC4Kernel/`:

```text
kernel-roundtrip.mlir
identity-roundtrip.mlir
predicate-roundtrip.mlir
fragment-ops-roundtrip.mlir
tmu-vdw-roundtrip.mlir
vpm-roundtrip.mlir
vdr-vpm-roundtrip.mlir
barrier-roundtrip.mlir
resource-roundtrip.mlir
```

### 21.2 Invalid diagnostics

Required invalid tests:

```text
invalid-forbidden-vector-op.mlir
invalid-forbidden-memref.mlir
invalid-forbidden-scf.mlir
invalid-forbidden-thread-id.mlir
invalid-index-type.mlir
invalid-vector-width.mlir
invalid-vector-i1-mask.mlir
invalid-sub32-type.mlir
invalid-kernel-arg-attrs.mlir
invalid-resource-metadata.mlir
invalid-predicate-nonnormalizable.mlir
invalid-vdw-noncontiguous-store.mlir
invalid-vdr-predicate.mlir
invalid-vpm-out-of-bounds.mlir
invalid-barrier-independent-vector.mlir
```

Each invalid test must check a deterministic semantic diagnostic, not only a parser failure.

### 21.3 Conversion lit tests

Required files under `compiler/test/Conversion/VC4KernelToSSAVC4/`:

```text
minimal-kernel.mlir
formal-args-launch-abi.mlir
identity-lowering.mlir
lane-range-lowering.mlir
predicate-tail-lowering.mlir
fragment-arith-lowering.mlir
fragment-rotate-reduce-lowering.mlir
tmu-load-fragment.mlir
vdw-store-fragment.mlir
vpm-read-write-fragment.mlir
vdr-load-to-vpm.mlir
vdw-store-vpm-fragment.mlir
barrier-lowering.mlir
control-flow-block-args.mlir
reject-forbidden-surface-ops.mlir
reject-direct-scheduled-vc4.mlir
```

### 21.4 Hardware fixtures

At least these executable hardware fixtures must be added under `compiler/test/CodeGen/VC4Kernel/Hardware/Run/` or the equivalent final location:

```text
vector_store_smoke_vc4kernel
program_id_writeback_vc4kernel
tmu_load_saxpy_vc4kernel
vdw_tail_store_vc4kernel
vpm_roundtrip_vc4kernel
vdr_to_vpm_roundtrip_vc4kernel
vpm_to_global_vdw_vc4kernel
fragment_reduce_sum_vc4kernel
control_flow_tail_block_args_vc4kernel
qpu_barrier_syncthreads_vc4kernel
```

Every hardware fixture must:

```text
- generate candidate artifacts from vc4kernel input
- lower through vc4kernel -> ssavc4 -> scheduled vc4
- assemble/build/run on real hardware
- compare copied-back device output to a CPU reference
- check sentinels for inactive-store preservation where stores are predicated
- check zero-fill behavior where loads are predicated
- not use reference QASM
- not fake VC4_TEST_RESULT
- not compute the output on the host harness instead of the device
```

### 21.5 Static anti-shortcut tests

The test suite must scan for and reject:

```text
- direct VC4KernelToVC4 conversion pass
- producer lowering in Stage 1
- executable sub-32 lowering
- vc4kernel.tile_* surface op definitions
- vc4kernel.thread_id
- vector dialect ops allowed by verifier
- memref/tensor/scf/tt/gpu inside verified vc4kernel
- fixture-name special casing in conversion
- comments or dummy literals added solely to satisfy verifier scans
```

---

## 22. Implementation checklist for the copy/prune/adapt step

After copying `vc4tile` to `vc4kernel`, perform this checklist in order.

### 22.1 Namespace and build integration

```text
- Rename include/lib/test directories to VC4Kernel.
- Rename TableGen prefix VC4Tile -> VC4Kernel.
- Rename C++ namespace vc4tile -> vc4kernel.
- Register VC4Kernel dialect in vc4-opt.
- Add VC4Kernel dialect and conversion libraries to CMake.
- Add --verify-vc4kernel.
- Add --convert-vc4kernel-to-ssavc4.
```

### 22.2 Delete/prune old surface

Delete all old ergonomic surface op definitions, verifiers, conversion patterns, tests, and verifier allowances:

```text
tile_descriptor
tile_load
tile_store
copy_tile
tile_view
tile_subview
transpose_view
shared_tile_alloc
tile_fill
tile_broadcast
tile_add
tile_sub
tile_mul
tile_select
tile_reduce
row_reduce
warp_reduce
block_reduce
tile_dot
tile_contract
tile_matmul
surface_placeholder
```

Do not leave them as hidden aliases in `vc4kernel`.

### 22.3 Delete old surface attrs/types

Delete or exclude:

```text
layout attrs
role attrs
precision attrs
packing attrs
M5 tile descriptor types
storage/expressed/accumulator metadata attrs
```

### 22.4 Replace old names with final path names

Map implementation source material as follows:

```text
vc4tile.masked_load_global  -> vc4kernel.tmu_load_fragment
vc4tile.masked_store_global -> vc4kernel.vdw_store_fragment
vc4tile.shared_alloc        -> vc4kernel.vpm_alloc
vc4tile.shared_load         -> vc4kernel.vpm_read_fragment
vc4tile.shared_store        -> vc4kernel.vpm_write_fragment
vc4tile.shared_store_global -> vc4kernel.vdw_store_vpm_fragment
vc4tile.vdr_load_tile       -> vc4kernel.vdr_load_to_vpm
vc4tile.rotate              -> vc4kernel.fragment_rotate
vc4tile.reduce              -> vc4kernel.fragment_reduce
vc4tile.mask_all            -> vc4kernel.pred.full
vc4tile.tail_mask           -> vc4kernel.pred.tail
```

### 22.5 Replace `vector.broadcast`

Any copied `vector.broadcast` or `vector.splat` usage inside `vc4kernel` must become:

```text
vc4kernel.splat
```

The verifier must reject all `vector.*` operations.

### 22.6 Remove `thread_id`

Do not port `vc4tile.thread_id`.

If copied accidentally, delete it and add invalid tests proving `vc4kernel.thread_id` is not accepted.

### 22.7 Convert raw masks to predicate type

Replace raw `vector<16xi1>` mask results/operands with:

```text
!vc4kernel.pred<16>
```

Ensure all consumers are updated and tests reject `vector<16xi1>` masks.

### 22.8 Tighten ODS constraints

No final `vc4kernel` op should use unconstrained `AnyType` when a precise constraint exists.

Use precise constraints for:

```text
i1
i32
f32
vector<16xi32>
vector<16xf32>
!vc4kernel.pred<16>
!vc4kernel.vpm_tile
```

### 22.9 Conversion cleanup

The new conversion must be named and scoped as:

```text
VC4KernelToSSAVC4
```

It must not contain surface pipeline logic such as:

```text
canonicalize-vc4tile-surface
plan-vc4tile-copies
surface_placeholder handling
copy_tile planning
```

If planning is needed, it belongs above `vc4kernel` in the future vector-to-vc4kernel pass.

---

## 23. Acceptance definition

`vc4kernel` Stage 1 is accepted only when all of these are true:

```text
1. The dialect exists as vc4kernel with final names and namespace.
2. The ODS op/type/attr inventory exactly matches this document.
3. The verifier rejects every forbidden dialect/type/op class listed here.
4. No old ergonomic vc4tile surface op exists in vc4kernel.
5. No vc4kernel.thread_id exists.
6. No vector dialect op is legal in vc4kernel.
7. No memref/tensor/scf/tt/gpu/linalg/producers are legal in vc4kernel.
8. Raw vector<16xi1> masks are replaced by !vc4kernel.pred<16>.
9. Global memory ops use explicit TMU/VDR/VDW path names.
10. VPM ops use !vc4kernel.vpm_tile and VPM-specific names.
11. Conversion is vc4kernel -> ssavc4 only.
12. No direct vc4kernel -> scheduled vc4 path exists.
13. Required lit tests pass.
14. Required hardware fixtures pass on real VC4 hardware.
15. check-vc4 remains green.
16. Anti-shortcut/integrity scans pass.
```

If any item fails, the dialect is not complete.

---

## 24. Future revisions explicitly outside this spec

The following are deliberately outside the initial final `vc4kernel` spec:

```text
- Triton TTIR parsing/registration/lowering
- IREE/Linalg lowering
- standard vector/memref/arith -> vc4kernel lowering implementation
- executable sub-32 precision
- quantization
- packed VPM lane modes
- arbitrary sparse predicates
- arbitrary scatter stores
- fragment_contract / matmul-specific planned op
- column-major/transposed layout algebra as a vc4kernel surface
- dynamic VPM layout beyond verified hardware modes
- multi-axis program_id support beyond axis 0
```

These may be added only by separate design documents and vertical hardware-proven implementation slices.

---

## 25. Final one-paragraph contract

`vc4kernel` is a strict VC4 target-kernel planning dialect below standard MLIR vector/memref/arith/scf/cf and above SSAVC4. It contains only kernel metadata, logical identity, 16-lane data fragments, structured VC4 predicates, explicit TMU/VDR/VPM/VDW memory path operations, VPM resource handles, barriers, scalar CFG, and target fragment arithmetic/reduction operations. It contains no old ergonomic tile surface, no memref, no vector dialect operations, no raw SCF, no Triton/IREE/producers, no thread_id, no vector<16xi1> masks, and no executable sub-32 precision. Its only valid lower path is `vc4kernel -> ssavc4 -> scheduled vc4 -> artifacts/runtime/hardware`.
