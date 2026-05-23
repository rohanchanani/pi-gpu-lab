# SSAVC4 IR Design Document for M3 — Post-cleanup Revision

**Status:** revised for the post-M2, post-structured-`vc4` cleanup repository state.  
**Purpose:** define the target-specific SSA VC4 IR for M3 and specify how it lowers into the existing scheduled `vc4` dialect.  
**Intended reader:** a junior engineer or coding agent implementing M3 from the current repo baseline.  
**Primary rule:** do not disturb the M2-proven scheduled `vc4` artifact sink.

---

## 0. Context and locked project state

M2 completed the backend from already-scheduled `vc4` QPU kernels to real Raspberry Pi VideoCore IV artifacts and hardware execution. The scheduled `vc4` dialect is now the locked backend sink. It emits QASM, manifest v2 metadata, `layout.json`, generated `kernel_launch.c/h`, shader C/H arrays, and uses the libpi-backed VC4 runtime for allocation, copy, raw SRQ scheduling, independent-vector scheduling, cooperative-block scheduling, and hardware execution.

After M2, the legacy structured form of the `vc4` dialect was removed. Active `vc4` is now scheduled-sink-only. The active `vc4` operation surface is:

```text
vc4.module
vc4.func
vc4.qpu.ldi
vc4.qpu.sema
vc4.qpu.bundle
vc4.qpu.branch
```

plus the existing launch/resource metadata and the live QPU attrs/enums required by these scheduled operations. The old structured `vc4` ops are gone from the active dialect and must not be resurrected:

```text
vc4.uniform.*
vc4.tmu.*
vc4.vpm.*
vc4.dma.*
vc4.sfu.*
vc4.return
vc4.program_end
vc4.async.wait
vc4.enqueue_qpu
vc4.reserve_qpu
vc4.v3d.*
function_form<structured>
```

M2 was also migrated to generic milestone automation. Future milestones should be driven by:

```text
pro_scripts/vc4_milestone_verifier.py
pro_scripts/vc4_milestone_autorun.py
pro_scripts/vc4_milestone_resume.py
pro_scripts/vc4_milestone_failure_router.py
pro_scripts/milestones/<milestone>.json
```

with milestone-specific files:

```text
pro_scripts/<milestone>_worklist.json
pro_scripts/<milestone>_verifications.json
pro_scripts/<milestone>_context_profiles.json
pro_scripts/prompts/<milestone>/**
```

M3 must be built on this cleaned state.

---

## 1. One-sentence design

`ssavc4` is a target-specific MLIR SSA machine IR for VideoCore IV QPU programs. It models the same low-level QPU/VPM/TMU/VDW/SFU/semaphore/runtime-resource semantics that scheduled `vc4` ultimately emits, but it does so with SSA values, explicit effects, virtual resources, and no physical register addresses, no fixed final instruction order, and no branch-distance scheduling. It lowers to scheduled `vc4.qpu.*` through instruction selection, out-of-SSA, register allocation, conservative scheduling, bundling, hazard insertion, branch layout, and metadata preservation.

Canonical pipeline after M4 and later producer work:

```text
Triton TTIR / IREE late executable IR / producer kernel IR
  ↓ future producer adapters, after M4
vc4tile dialect
  ↓ M4, not part of M3
ssavc4 dialect
  ↓ M3
scheduled vc4 dialect
  ↓ already completed in M2
QASM + kernel_launch.c/h + shader arrays + libpi runtime execution
```

M3 is **not** producer lowering from `gpu`, Triton, IREE, or any other frontend IR. M3 is **only** `ssavc4 -> scheduled vc4` plus the SSAVC4 IR definition and tests. M4 is the VC4 Tile dialect (`vc4tile`) and `vc4tile -> ssavc4` lowering; producer lowering into `vc4tile` is later work.

---

## 2. Why SSAVC4 exists

The scheduled `vc4` dialect is close to QASM. It exposes physical read/write register addresses, mux selections, ADD/MUL opcodes, pack/unpack modes, QPU signals, load-immediate forms, branch immediates, branch delay slots, semaphores, VPM/VDW/TMU/SFU setup words, waits, thread-end epilogues, launch ABI metadata, and resource metadata.

That is exactly right for the post-scheduling artifact boundary, but it is too physical for direct lowering from upstream IR. M4 must not lower directly from `gpu` or other producer IR to scheduled `vc4`; it introduces `vc4tile` so producer-like tile/kernel structure can be normalized before SSAVC4 lowering. Direct producer-to-`vc4` lowering would have to make these decisions too early:

```text
physical register allocation
A/B register-file placement
accumulator usage
ADD/MUL pipeline choice
bundle order
branch immediates
branch delay slots
TMU latency/wait placement
VPM/VDW setup sequencing
semaphore ordering
cooperative resource layout
thread-end epilogue shape
```

SSAVC4 separates target semantics from physical scheduling decisions. It is analogous to a target-specific pre-register-allocation machine SSA IR. Scheduled `vc4` remains analogous to post-register-allocation, post-scheduling machine instructions.

---

## 3. Hard scope boundaries

### 3.1 M3 goals

M3 must implement:

1. A new `ssavc4` dialect, separate from `vc4`.
2. SSAVC4 parser/printer/verifiers/tests.
3. SSA value operations for low-level QPU dataflow.
4. Explicit side-effect/resource modeling for ordered hardware operations.
5. A lowering pass from `ssavc4` to scheduled `vc4`.
6. End-to-end M3 fixtures that start from SSAVC4 input and still emit M2-compatible artifacts.
7. Cumulative regression: M2 scheduled fixtures must continue to pass.

### 3.2 M3 non-goals

M3 must not:

1. Lower directly from MLIR `gpu` or another producer IR to scheduled `vc4`.
2. Reintroduce legacy structured `vc4` ops.
3. Replace or weaken the scheduled `vc4` sink.
4. Change `VC4ArtifactEmitter.cpp` except for strictly necessary generic support that preserves all M2 tests. In normal M3 slices, the emitter should not need changes.
5. Branch on fixture/public names in lowering, emitter, support runner, or runtime.
6. Use physical QPU number as normal logical identity.
7. Add high-level algorithmic ops such as `ssavc4.matmul`, `ssavc4.conv`, or `ssavc4.shared_tile_transpose`.
8. Mark uniform/TMU/VPM/VDW/DMA/SFU/semaphore/mutex/barrier/thread-end operations as pure.
9. Require user-facing physical register addresses in SSAVC4.
10. Implement full spilling, aggressive ADD/MUL bundling, or useful branch-delay-slot filling in the first M3 slices.
11. Modify reference bundles to pass M3. Reference bundles remain ground truth/historical fixtures.
12. Treat `vpm_setup_clobber` as an M3 acceptance fixture. It remains post-M2 hardware characterization.

---

## 4. Dialect identity

Use:

```text
Dialect mnemonic: ssavc4
C++ namespace:    ::mlir::ssavc4
```

Do not implement SSAVC4 as `form = structured` inside `vc4`. That path was intentionally removed. Keeping SSAVC4 separate prevents ambiguity:

```text
ssavc4 = pre-RA / pre-scheduling SSA target machine IR
vc4    = post-RA / post-scheduling QASM-near sink IR
```

Illegal mixing should be obvious. `vc4-codegen` consumes scheduled `vc4`, not SSAVC4. The SSAVC4 lowering pass produces scheduled `vc4`, and then the existing M2 artifact pipeline runs unchanged.

---

## 5. Repository layout for M3 implementation

Add:

```text
compiler/include/vc4/Dialect/SSAVC4/IR/CMakeLists.txt
compiler/include/vc4/Dialect/SSAVC4/IR/SSAVC4Base.td
compiler/include/vc4/Dialect/SSAVC4/IR/SSAVC4Dialect.td
compiler/include/vc4/Dialect/SSAVC4/IR/SSAVC4Types.td
compiler/include/vc4/Dialect/SSAVC4/IR/SSAVC4Attrs.td
compiler/include/vc4/Dialect/SSAVC4/IR/SSAVC4Ops.td
compiler/include/vc4/Dialect/SSAVC4/IR/SSAVC4Ops.h
compiler/include/vc4/Dialect/SSAVC4/IR/SSAVC4Types.h

compiler/lib/Dialect/SSAVC4/IR/CMakeLists.txt
compiler/lib/Dialect/SSAVC4/IR/SSAVC4Dialect.cpp
compiler/lib/Dialect/SSAVC4/IR/SSAVC4Ops.cpp
compiler/lib/Dialect/SSAVC4/IR/SSAVC4Types.cpp

compiler/include/vc4/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.h
compiler/lib/Conversion/SSAVC4ToVC4/CMakeLists.txt
compiler/lib/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.cpp

compiler/docs/ssavc4-ir-design.md
compiler/test/Dialect/SSAVC4/**
compiler/test/Conversion/SSAVC4ToVC4/**
compiler/test/CodeGen/SSAVC4/**
```

Edit:

```text
compiler/include/vc4/Dialect/CMakeLists.txt
compiler/lib/Dialect/CMakeLists.txt
compiler/include/vc4/CMakeLists.txt
compiler/lib/CMakeLists.txt
compiler/tools/vc4-opt/vc4-opt.cpp
compiler/test/CMakeLists.txt
```

`vc4-opt` must register both dialects:

```cpp
registry.insert<mlir::vc4::VC4Dialect,
                mlir::ssavc4::SSAVC4Dialect>();
```

It must register the conversion pass:

```text
--convert-ssavc4-to-vc4
```

M3 must not require `vc4-codegen` to consume SSAVC4 directly. The normal path is:

```bash
vc4-opt --convert-ssavc4-to-vc4 input.ssavc4.mlir -o scheduled.vc4.mlir
vc4-codegen scheduled.vc4.mlir --emit-bundle <bundle-dir>
```

A convenience wrapper script may run those two steps, but the artifact emitter remains scheduled-`vc4` only.

---

## 6. Relationship to current `vc4` enums, attrs, resources, and metadata

### 6.1 Reuse only live scheduled-QPU VC4 attrs/enums

The old design assumed many structured `vc4` attrs still existed, such as TMU/VPM/DMA/SFU descriptor attrs. After cleanup, they do not. Therefore:

1. SSAVC4 may reuse **only live `vc4` attrs/enums** that are still part of the active scheduled sink.
2. SSAVC4 must **not** depend on removed structured `vc4` attrs/enums.
3. If SSAVC4 needs descriptor/domain attrs for TMU, VPM, DMA, SFU, or VDW, define them in the `ssavc4` dialect.

Expected live reusable VC4 attr domains include the scheduled-QPU domains, such as:

```text
#vc4.add_opcode<...>
#vc4.mul_opcode<...>
#vc4.cond<...>
#vc4.qpu_signal<...>
#vc4.qpu_mux<...>
#vc4.load_imm_mode<...>
#vc4.regfile_a_unpack_mode<...>
#vc4.regfile_a_pack_mode<...>
#vc4.r4_unpack_mode<...>
#vc4.mul_pack_mode<...>
#vc4.semaphore_mode<...>
#vc4.branch_cond<...>
#vc4.execution_domain<qpu>
#vc4.function_form<scheduled>
#vc4.threading_mode<...>
#vc4.builtin_kind<...>
```

Before implementation, Codex/engineers must inspect the current live files:

```text
compiler/include/vc4/Dialect/VC4/IR/VC4Enums.td
compiler/include/vc4/Dialect/VC4/IR/VC4OpEnums.td
compiler/include/vc4/Dialect/VC4/IR/VC4Attrs.td
```

and use only attrs that actually exist. Do not trust the pre-cleanup structured-VC4 design.

### 6.2 Define SSAVC4 attrs for removed structured concepts

Add SSAVC4 attrs/enums only for concepts SSAVC4 truly needs. M3 v1 should keep these minimal.

Recommended SSAVC4 attr domains:

```text
#ssavc4.tmu_unit<tmu0 | tmu1>
#ssavc4.tmu_mode<direct>
#ssavc4.tmu_read_part<raw32>
#ssavc4.sfu_kind<recip | rsqrt | exp | log>        // only if SFU is implemented in M3
#ssavc4.vpm_orientation<horizontal | vertical>
#ssavc4.vpm_lane_mode<packed | laned>
#ssavc4.vpm_elem_width<w8 | w16 | w32>
#ssavc4.vdw_store_serialize<none | mutex>
#ssavc4.flag_kind<sub | zero_test | compare>
```

Do not define broad texture/cubemap/DMA descriptor attr sets unless a slice actually needs them. TMU texture/cubemap paths are deferred.

### 6.3 Reuse launch/resource dictionaries verbatim

`ssavc4.func` must carry the same metadata dictionaries that the M2 scheduled emitter already understands:

```text
"vc4.launch_abi" = { ... }
"vc4.resource" = { ... }
```

The conversion pass copies these dictionaries unchanged onto the lowered scheduled `vc4.func`.

Reason: M2 already established the launch ABI and resource metadata contract. SSAVC4 should feed that path, not fork it.

`vc4.launch_abi` is also the source of meaning for physical uniform slots. `ssavc4.uniform.read` reads the uniform stream; it does not by itself decide whether a slot is a user kernel argument or runtime/scheduler metadata. Kernel arguments are described by `vc4.launch_abi.args[]`. Runtime builtins such as logical request, logical block, logical warp, cooperative VPM/semaphore allocation, resident request, and spill-frame metadata are described by `vc4.launch_abi.builtins[]`.

`ssavc4.element_number` remains outside `vc4.launch_abi`: it is a pure hardware lane identity value derived from the QPU register state, not runtime-packed uniform metadata. Runtime logical identity builtins must describe software launch metadata and must not be aliases for the physical QPU number.

### 6.4 Shared side-effect resources

`VC4SideEffects.h` intentionally remains as shared VC4 hardware-resource infrastructure. SSAVC4 may use those resource classes if they still exist and fit the desired semantics.

Do not interpret a resource class as evidence that the old structured `vc4` op surface still exists. The resources are shared hardware-effect names; the old structured ops are gone.

---

## 7. Type system

### 7.1 Ordinary data types

Use MLIR builtin scalar/vector types for ordinary data:

```text
i32
f32
vector<16xi32>
vector<16xf32>
```

Do not introduce `!ssavc4.v16f32` or `!ssavc4.v16i32` in M3. The conceptual example:

```mlir
%z = ssavc4.fadd %x, %y : !ssavc4.v16f32
```

must be implemented as:

```mlir
%z = ssavc4.alu.add %x, %y {opcode = #vc4.add_opcode<fadd>}
  : vector<16xf32>
```

### 7.2 Allowed value shapes

Define helper predicates in `SSAVC4Ops.cpp`:

```cpp
bool isSSAVC4Scalar32(Type t);
bool isSSAVC4Vector16(Type t);
bool isSSAVC4IntCarrier(Type t);
bool isSSAVC4FloatCarrier(Type t);
bool isSSAVC4ValueType(Type t);
bool haveSameSSAVC4Shape(Type a, Type b);
bool haveSameSSAVC4ElementDomain(Type a, Type b);
```

Rules:

```text
i32                 scalar integer carrier
f32                 scalar float carrier
vector<16xi32>      16-lane integer carrier
vector<16xf32>      16-lane float carrier
```

No other vector length is legal inside a lowerable `ssavc4.func`.

Reject these in lowerable functions unless an explicit canonicalization pass has already converted them:

```text
i1, i8, i16, i64, f16, f64, index, memref, tensor, gpu dialect types
```

### 7.3 Custom SSAVC4 types

Add only:

```text
!ssavc4.async.token
!ssavc4.tmu.desc
!ssavc4.vpm.desc
!ssavc4.flags
```

Optional/deferred:

```text
!ssavc4.dma.desc
```

Do not implement `!ssavc4.dma.desc` until a slice truly needs generic DMA descriptor modeling. `ssavc4.vdw.store` should be the first practical global-store abstraction.

Meanings:

```text
!ssavc4.async.token   token for ordered async-like hardware queues, primarily TMU in M3 v1
!ssavc4.tmu.desc      opaque direct-mode TMU descriptor, minimal in M3 v1
!ssavc4.vpm.desc      opaque VPM read/write setup descriptor, minimal in M3 v1
!ssavc4.flags         SSA representation of branch/predicate flags
```

`!ssavc4.flags` is not a physical hardware flag register. It is an SSA pseudo-value. Lowering materializes it as a scheduled flag-setting ADD-pipe operation placed close to the branch or predicated use.

### 7.4 Flags restrictions

For M3 v1:

1. `!ssavc4.flags` values must be single-use.
2. `!ssavc4.flags` values must not be function arguments.
3. `!ssavc4.flags` values must not be block arguments.
4. `!ssavc4.flags` values must not be returned.
5. `!ssavc4.flags` values must not be stored in any token/descriptor.
6. If a flags value has multiple uses, the lowering must either reject it or explicitly clone/rematerialize the flag-setting operation near each use.

Reason: VC4 flags are transient hardware condition state, not ordinary storage.

---

## 8. Function and module model

### 8.1 `ssavc4.module`

`ssavc4.module` should mirror the role of `vc4.module` but contain SSAVC4 functions. It is a target module, not a generic host module.

Required properties:

```text
symbol table
isolated from above
one body region
no implicit host runtime behavior
```

### 8.2 `ssavc4.func`

`ssavc4.func` is the pre-scheduled target function.

Required attributes:

```text
sym_name
function_type
kernel, when launchable
threading, if needed for parity with vc4.func metadata
"vc4.launch_abi", for launchable kernels
"vc4.resource", for kernels needing resource scheduling metadata
```

Do not add `function_form<structured>`. SSAVC4 is already the separate pre-scheduled dialect.

### 8.3 Region/terminator policy

For M3 v1, choose the simplest policy that is robust in MLIR:

1. `ssavc4.func` has a body region containing ordinary blocks.
2. Non-external functions must terminate with `ssavc4.return` or `ssavc4.thread_end`, depending on whether the function is a helper or a kernel.
3. Launchable QPU kernels must ultimately lower to a scheduled thread-end epilogue, not to a host return.
4. If all M3 kernels are single-region QPU programs, `ssavc4.thread_end` may be required for launchable kernels.

Do not confuse `ssavc4.return` with removed `vc4.return`. If implemented, `ssavc4.return` is a new op in a different dialect.

### 8.4 Metadata copying

Lowering copies these from `ssavc4.func` to generated `vc4.func`:

```text
kernel
threading
"vc4.launch_abi"
"vc4.resource"
sym_name, possibly preserved exactly
function_type, adjusted only if scheduled vc4 requires zero SSA results
```

The scheduled `vc4.func` must be:

```text
domain = #vc4.execution_domain<qpu>
form   = #vc4.function_form<scheduled>
```

---

## 9. Effect model

### 9.1 Principle

Pure SSA arithmetic/value-shape operations must be marked pure.

Hardware-state operations must implement `MemoryEffectOpInterface` and must not be marked pure. Effects or explicit token operands/results are mandatory so MLIR optimizations do not illegally delete or reorder uniform reads, TMU requests, VPM/VDW operations, semaphores, mutexes, barriers, or thread termination.

### 9.2 Effect/resource table

| Op category | Pure? | Required ordering/effects |
|---|---:|---|
| `ssavc4.load_imm` | yes | none |
| `ssavc4.element_number` | yes | none |
| `ssavc4.splat` | yes | none |
| `ssavc4.mov` | yes | none |
| `ssavc4.alu.add` | yes | none |
| `ssavc4.alu.mul` | yes | none |
| `ssavc4.pack` | yes | none |
| `ssavc4.unpack` | yes | none |
| `ssavc4.rotate` | yes | none |
| `ssavc4.make_flags` | yes, with restrictions | none; but lowering restrictions apply |
| `ssavc4.tmu.desc` | yes | none |
| `ssavc4.vpm.desc` | yes | none |
| `ssavc4.uniform.read` | no | read uniform stream |
| `ssavc4.uniform.seek` | no | write/advance uniform stream; may be rejected in M3 v1 |
| `ssavc4.tmu.request` | no | write TMU request resource; read main memory |
| `ssavc4.tmu.read` | no | read TMU receive resource; ordered by token |
| `ssavc4.sfu.issue` | no | write SFU resource |
| `ssavc4.sfu.read` | no | read SFU resource |
| `ssavc4.vpm.read` | no | read VPM read FIFO/resource |
| `ssavc4.vpm.write` | no | write VPM write FIFO/resource |
| `ssavc4.vdw.store` | no | writes VPM/VDW/main memory path; ordered |
| `ssavc4.sema.acquire` | no | semaphore read/acquire effect |
| `ssavc4.sema.release` | no | semaphore write/release effect |
| `ssavc4.mutex.acquire` | no | mutex acquire effect |
| `ssavc4.mutex.release` | no | mutex release effect |
| `ssavc4.barrier` | no | semaphore/cooperative ordering effect |
| `ssavc4.thread_end` | no | thread-control termination effect |

M3 v1 should prefer token ordering where resource effects are not enough to force correct local order. For example, `ssavc4.tmu.request` returns a token consumed by `ssavc4.tmu.read`; `ssavc4.vdw.store` may return a token or may be modeled as an effect-only terminable sequence, depending on implementation simplicity.

---

## 10. Logical execution model

### 10.1 Independent-vector mapping

Normal independent-vector kernels follow this CUDA-like mapping:

```text
one runtime request = one QPU warp
one QPU warp = 16 vector lanes
ELEMENT_NUMBER = lane id
logical_request = warp/request id supplied by runtime/uniforms
num_qpus / total_requests = runtime-supplied launch geometry
```

SSAVC4 must not read physical QPU number for logical identity.

### 10.2 Cooperative-block mapping

Cooperative kernels follow:

```text
one block = several resident QPU warp requests
logical_warp_id = warp id inside block
logical_block_id = block id
warps_per_block = runtime-supplied cooperative block width
vpm_base_row = runtime-supplied per-resident-block VPM allocation
semaphore_base = runtime-supplied per-resident-block semaphore allocation
```

These values are represented as uniform reads / launch ABI builtins in SSAVC4. They are not physical-QPU reads.

The phrase "uniform reads / launch ABI builtins" means that the physical read is `ssavc4.uniform.read`, while the semantic category comes from the `vc4.launch_abi` builtin entry. These logical identity values are runtime/scheduler metadata and are not user kernel arguments.

### 10.3 Physical QPU dependency ban

Normal lowering must not use physical QPU identity for program semantics. Physical-QPU fixtures such as `vpm_setup_clobber` are hardware characterization only. They are not M3 acceptance fixtures.

---

## 11. Core pure operations

### 11.1 `ssavc4.load_imm`

Syntax examples:

```mlir
%c = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
%v = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
%p = ssavc4.load_imm <per_elem_u2> {values = dense<[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]> : vector<16xi32>} : vector<16xi32>
```

Rules:

1. Pure.
2. Result must be an allowed SSAVC4 value type.
3. `splat32` requires one 32-bit value attribute.
4. `per_elem_i2` requires exactly 16 values in `[-2, 1]`, result `vector<16xi32>`.
5. `per_elem_u2` requires exactly 16 values in `[0, 3]`, result `vector<16xi32>`.
6. `f32` values are represented through their 32-bit carrier bits during lowering.

Lowering:

1. Fold into small immediate if legal at a use.
2. Otherwise emit scheduled `vc4.qpu.ldi`.
3. Preserve literal semantics exactly; do not invent host-side constants.

### 11.2 `ssavc4.element_number`

Syntax:

```mlir
%lane = ssavc4.element_number : vector<16xi32>
```

Rules:

1. Pure.
2. Result type is `vector<16xi32>`.
3. Represents QPU `ELEMENT_NUMBER`, the lane id.
4. Not a request id.
5. Not a physical QPU id.

Lowering:

1. Emit scheduled read of hardware element-number source.
2. Allocate the result as a normal virtual value.

### 11.3 `ssavc4.splat`

Syntax:

```mlir
%v = ssavc4.splat %x : i32 -> vector<16xi32>
%v = ssavc4.splat %a : f32 -> vector<16xf32>
```

Rules:

1. Pure.
2. Input must be scalar `i32` or `f32`.
3. Result must be corresponding `vector<16x...>`.
4. Lowering may be a no-op if the QPU representation is already lane-replicated.

### 11.4 `ssavc4.mov`

Syntax:

```mlir
%y = ssavc4.mov %x : vector<16xf32>
```

Rules:

1. Pure.
2. Input/result types identical.
3. Canonicalization may erase it.
4. Lowering may reintroduce moves for register constraints.

### 11.5 `ssavc4.alu.add`

Syntax:

```mlir
%z = ssavc4.alu.add %x, %y {opcode = #vc4.add_opcode<fadd>} : vector<16xf32>
%z = ssavc4.alu.add %x {opcode = #vc4.add_opcode<clz>} : vector<16xi32>
```

Rules:

1. Pure.
2. Opcode must be a live scheduled `vc4` ADD opcode attr.
3. Binary ops require same shape and element domain.
4. Unary ops must use exactly one operand.
5. `ftoi` maps float carriers to integer carriers with same shape.
6. `itof` maps integer carriers to float carriers with same shape.
7. M3 v1 supports only opcodes already proven in scheduled VC4 fixtures/lit tests; unsupported live opcodes may parse but lowering may reject with a precise diagnostic.

Lowering:

1. Emit ADD-pipe scheduled operation.
2. M3 v1 may emit one active pipe per bundle.
3. Later bundling may combine independent ADD and MUL ops.

### 11.6 `ssavc4.alu.mul`

Syntax:

```mlir
%z = ssavc4.alu.mul %x, %y {opcode = #vc4.mul_opcode<fmul>} : vector<16xf32>
```

Rules:

1. Pure.
2. Opcode must be a live scheduled `vc4` MUL opcode attr.
3. Operands/result must have compatible shapes and domains.
4. M3 v1 supports only opcodes already proven in scheduled VC4 fixtures/lit tests.

Lowering:

1. Emit MUL-pipe scheduled operation.
2. M3 v1 may keep the ADD pipe inactive.

### 11.7 `ssavc4.pack`, `ssavc4.unpack`, `ssavc4.rotate`

These are pure value transforms. They model QPU pack/unpack/rotate semantics without physical register placement.

Rules:

1. Use only live scheduled `vc4` pack/unpack attrs.
2. Do not introduce i8/i16 storage types.
3. Carrier types remain `i32`, `f32`, `vector<16xi32>`, or `vector<16xf32>`.
4. Rotate input/result must be vector<16x...>.
5. Immediate rotate amount must be `[0, 15]`.

Lowering:

1. Fuse pack/unpack into producing/consuming scheduled ops where legal.
2. Otherwise emit a move with pack/unpack.
3. Rotate uses small-immediate rotate selectors or R5 rotate path as appropriate.
4. Existing scheduled verifiers must catch illegal rotate/register constraints.

---

## 12. Uniform operations

### 12.1 `ssavc4.uniform.read`

Syntax:

```mlir
%x = ssavc4.uniform.read {name = "x", uniform_index = 0 : i32} : i32
%alpha = ssavc4.uniform.read {name = "alpha", uniform_index = 2 : i32} : f32
```

Rules:

1. Effectful: reads uniform stream.
2. Not pure.
3. Result type must be `i32` or `f32` in M3 v1.
4. `uniform_index` is required for launch ABI reads.
5. `name`, if present, must correspond to an argument or builtin in `"vc4.launch_abi"`.
6. ABI uniform indices must be dense and match `uniform_words_per_qpu`.
7. Uniform reads for ABI values must appear in entry block before non-uniform effectful ops in M3 v1.
8. `ELEMENT_NUMBER` is not read through this op. Use `ssavc4.element_number`.

`ssavc4.uniform.read` is the physical uniform-stream read operation. The meaning of the slot is defined by `vc4.launch_abi`: entries in `args[]` are user/caller-supplied kernel arguments, and entries in `builtins[]` are runtime/scheduler-supplied metadata. Do not model `ssavc4.element_number` as a uniform read or as a launch ABI builtin.

Lowering:

1. Emit scheduled read of `unif`.
2. Preserve read order.
3. Allocate destination as a virtual value.

### 12.2 `ssavc4.uniform.seek`

Deferred. It may parse in a later slice, but M3 v1 lowering may reject it unless a fixture requires it.

---

## 13. Memory and peripheral operations

### 13.1 Design principle

SSAVC4 memory/peripheral operations are target-specific. They are not generic `memref.load` / `memref.store` and not high-level GPU shared memory abstractions.

The initial supported path should be driven by the M2 fixtures:

```text
VDW/global vector stores first
TMU direct loads next
reductions/rotates after that
cooperative VPM/semaphore barriers later
```

### 13.2 `ssavc4.vdw.store`

This is the first practical 1:N low-level memory op for M3. It is target-specific shorthand for the VC4 VPM→VDW store path.

Syntax sketch:

```mlir
ssavc4.vdw.store %addr, %value {
  elem_bytes = 4 : i32,
  active_lanes = 16 : i32,
  vpm_row = 0 : i32,
  serialize = #ssavc4.vdw_store_serialize<mutex>
} : i32, vector<16xf32>
```

Rules:

1. Effectful.
2. `%addr` is scalar `i32` bus/device address.
3. `%value` is `vector<16xi32>` or `vector<16xf32>` in M3 v1.
4. `active_lanes` must be `1..16`.
5. `elem_bytes` must be `4` in M3 v1.
6. `vpm_row` must be within target VPM capacity.
7. `serialize = mutex` means lowering must wrap the VPM/VDW sequence with mutex acquire/release if required by the known-good scheduled store path.
8. This op is not a generic memory store and must not hide host memory semantics.

Lowering:

1. Write VPM setup for vector write staging.
2. Write the vector value to VPM.
3. Wait/serialize as required by the known-good M2 scheduled store path.
4. Write VDW setup for horizontal/vertical store as required.
5. Write VDW address.
6. Wait for VDW completion.
7. Release mutex if acquired.

M3 v1 must follow known-good M2 patterns from scheduled fixtures such as `global_store_coalesced_multi` / `saxpy_full`; do not invent a new store protocol.

### 13.3 `ssavc4.tmu.request` and `ssavc4.tmu.read`

M3 v1 supports direct TMU loads only.

Syntax sketch:

```mlir
%tok = ssavc4.tmu.request %addr {
  unit = #ssavc4.tmu_unit<tmu0>,
  mode = #ssavc4.tmu_mode<direct>
} : vector<16xi32> -> !ssavc4.async.token

%x = ssavc4.tmu.read %tok {
  unit = #ssavc4.tmu_unit<tmu0>,
  part = #ssavc4.tmu_read_part<raw32>
} : !ssavc4.async.token -> vector<16xf32>
```

Rules:

1. Both ops are effectful.
2. Direct-mode address type is `i32` or `vector<16xi32>`.
3. Request produces a token.
4. Read consumes a token from a matching unit.
5. M3 v1 supports `tmu0` and `raw32` first.
6. `tmu1`, packed read parts, texture/cubemap descriptors, and TMU overlap optimizations are deferred unless a slice explicitly enables them.

Lowering:

1. Request lowers to scheduled TMU parameter/address write.
2. Read lowers to scheduled `ldtmu0`/receive sequence.
3. Scheduler must preserve `r4` lifetime after TMU receive.
4. Conservative non-overlapped TMU scheduling is acceptable in M3 v1.

### 13.4 `ssavc4.sfu.issue` and `ssavc4.sfu.read`

Optional until a slice needs `sfu_recip`.

Rules:

1. Effectful.
2. `issue` writes SFU resource.
3. `read` reads SFU result.
4. Lowering must enforce SFU latency/hazard spacing conservatively.

### 13.5 `ssavc4.vpm.read` and `ssavc4.vpm.write`

Do not implement a broad VPM descriptor language early. Implement only the subset needed by:

```text
vdw.store
warp reduction/prefix fixtures
cooperative barrier/shared VPM fixtures
```

If descriptor types are added, they are SSAVC4 descriptor types, not old `vc4.vpm.desc`.

### 13.6 Generic DMA descriptors

Defer `ssavc4.dma.desc`, `ssavc4.dma.start`, and `ssavc4.dma.wait` until M3 has a specific fixture that needs generic VDR/VDW DMA modeling beyond `ssavc4.vdw.store`.

---

## 14. Synchronization and cooperative operations

### 14.1 `ssavc4.sema.acquire` and `ssavc4.sema.release`

Syntax sketch:

```mlir
ssavc4.sema.release %sem : i32
ssavc4.sema.acquire %sem : i32
```

Rules:

1. Effectful.
2. `%sem` is scalar `i32` in M3 v1.
3. Constant semaphore IDs may lower directly to `vc4.qpu.sema` ids.
4. Dynamic semaphore base + offset may lower to a fixed id only if compile-time resolved by launch/resource metadata; otherwise reject in M3 v1.

### 14.2 `ssavc4.barrier`

Syntax sketch:

```mlir
ssavc4.barrier {
  arrive_offset = 0 : i32,
  go_offset = 1 : i32,
  depart_offset = 2 : i32,
  reset_offset = 3 : i32
}
```

Rules:

1. Effectful.
2. Only legal in a function with `"vc4.resource"` declaring cooperative-block scheduling and barrier use.
3. Requires `warps_per_block` and semaphore resource metadata.
4. M3 v1 may lower to the same reusable four-semaphore barrier pattern used by M2 cooperative fixtures.
5. Do not hardcode fixture names.

### 14.3 `ssavc4.mutex.acquire` / `ssavc4.mutex.release`

Use only where the known-good VPM/VDW path requires mutex serialization. Do not use mutex as a substitute for cooperative scheduling.

---

## 15. Control flow and branches

### 15.1 SSAVC4 branch ops

Add:

```text
ssavc4.br
ssavc4.cond_br
ssavc4.make_flags
```

Possible syntax:

```mlir
%flags = ssavc4.make_flags %x, %y {kind = #ssavc4.flag_kind<sub>} : vector<16xi32>
ssavc4.cond_br %flags, ^then, ^else {cond = #vc4.branch_cond<any_z_set>}
ssavc4.br ^next
```

Rules:

1. `ssavc4.make_flags` produces `!ssavc4.flags`.
2. `ssavc4.cond_br` consumes exactly one `!ssavc4.flags` value.
3. Flags single-use restrictions apply.
4. `cond_br` must use a live scheduled VC4 branch condition attr.
5. M3 v1 may support only direct block successors and no branch weights.
6. M3 v1 may reject irreducible CFGs.
7. M3 v1 may reject loops requiring spilling.

### 15.2 Branch lowering

The old draft hardcoded branch details that are too risky. M3 must instead derive exact scheduled branch fields from the current scheduled `vc4` verifier/emitter conventions and passing fixtures.

Required procedure:

1. Schedule all non-branch operations into a linear slot list.
2. Assign slot numbers.
3. Insert `vc4.qpu.branch` operations.
4. Fill each branch delay-slot region with exactly three scheduled QPU ops.
5. In M3 v1, all delay slots are scheduled nops.
6. Compute branch immediate after final layout.
7. Use the same branch field conventions as passing M2 scheduled fixtures.

Default M3 v1 branch shape should match existing scheduled fixtures unless current code says otherwise:

```text
relative = true
use_reg = false
raddr_a = 0
waddr_add = 31
waddr_mul = 30
3 nop delay-slot ops
```

Do not use the old `waddr_add = 39`, `waddr_mul = 39` assumption. Before implementing, inspect:

```text
compiler/include/vc4/Dialect/VC4/IR/VC4ScheduledQPUOps.td
compiler/lib/Dialect/VC4/IR/VC4Ops.cpp
compiler/lib/Target/VC4/VC4ArtifactEmitter.cpp
compiler/test/CodeGen/VC4/Emit/emit-qpu-branch-delay-slots.mlir
compiler/test/CodeGen/VC4/Hardware/Run/*/input.mlir
```

Acceptance for branch lowering:

1. Scheduled output passes the existing scheduled branch verifier.
2. Generated QASM branch labels/offsets assemble.
3. A fixture with tail control flow runs on hardware.

---

## 16. Lowering architecture

### 16.1 Pass name

```text
--convert-ssavc4-to-vc4
```

### 16.2 Lowering input/output

Input:

```text
ssavc4.module containing ssavc4.func kernels
```

Output:

```text
vc4.module containing scheduled vc4.func kernels with only vc4.qpu.* body ops
```

The output must be accepted by existing scheduled verifiers and `vc4-codegen --emit-bundle`.

### 16.3 Internal lowering pipeline

Implement the conversion pass in phases:

```text
1. Verify SSAVC4 launch/resource metadata.
2. Canonicalize simple pure ops where safe.
3. Build per-function SSA value graph.
4. Select each SSAVC4 op into a target instruction template or template sequence.
5. Assign virtual registers/value classes.
6. Perform simple liveness analysis.
7. Allocate physical locations, using spill-frame support when needed.
8. Schedule into linear QPU slots with hazards/nops.
9. Lay out branches and delay slots.
10. Emit scheduled vc4.qpu.* operations.
11. Copy metadata to generated vc4.func.
12. Run/require scheduled verifier pass.
```

### 16.4 Instruction templates

Each selected op lowers to a template sequence, not directly to `vc4.qpu.*` immediately.

Example template categories:

```text
ADD_ALU(opcode, operands, result)
MUL_ALU(opcode, operands, result)
LDI(mode, value, result)
READ_UNIF(result)
READ_ELEM_NUM(result)
TMU_REQUEST(unit, addr, token)
TMU_READ(unit, token, result)
VPM_WRITE_SETUP(...)
VPM_WRITE(value)
VDW_SETUP(...)
VDW_ADDR(addr)
WAIT(kind)
SEMA_ACQUIRE(id)
SEMA_RELEASE(id)
BRANCH(cond, target)
THREND
NOP
```

Templates then become scheduled `vc4.qpu.*` with physical addresses.

### 16.5 Register allocation v1

M3 v1 allocator is intentionally conservative.

Rules:

1. No spilling.
2. Reject if live ranges exceed available safe registers.
3. Prefer ordinary regfile locations for long-lived values.
4. Use accumulators for short-lived values, hardware results, and operations that require them.
5. Preserve `r4` around TMU/SFU receives.
6. Preserve required VPM/VDW/TMU setup temporaries until their side-effecting use.
7. Avoid scheduling writes to peripheral/stall addresses except through intended template sequences.
8. Let existing scheduled verifier reject illegal physical combinations.

This is allowed to be inefficient. M3 v1 correctness matters more than optimality.

### 16.6 Scheduling v1

Rules:

1. One active pipeline per bundle by default.
2. Insert nops conservatively for hazards.
3. Do not attempt useful delay-slot filling.
4. Do not attempt overlapping TMU unless a later slice explicitly adds it.
5. Do not attempt aggressive ADD/MUL pairing.
6. Use existing scheduled verifier passes as the safety net.

### 16.7 Metadata preservation

The lowering pass must preserve:

```text
vc4.launch_abi public_name
vc4.launch_abi code_symbol
vc4.launch_abi uniform_words_per_qpu
vc4.launch_abi args
vc4.launch_abi builtins
vc4.launch_abi tail_policy
vc4.resource schedule_mode
vc4.resource uses_barrier
vc4.resource uses_shared_vpm
vc4.resource require_full_block_residency
vc4.resource warps_per_block_max
vc4.resource semaphores_per_block
vc4.resource shared_vpm_bytes / rows fields
```

Do not recalculate launch ABI in the lowering pass except for validation. The existing emitter/runtime path owns launch packing and resource scheduling.

---

## 17. Operation surface for M3 v1

Do not implement every possible op in one early slice. Implement only what the slice requires.

### 17.1 M3 v1 minimum operation set

For the first real hardware-output path:

```text
ssavc4.module
ssavc4.func
ssavc4.thread_end
ssavc4.load_imm
ssavc4.uniform.read
ssavc4.element_number
ssavc4.splat
ssavc4.mov
ssavc4.alu.add
ssavc4.alu.mul
ssavc4.vdw.store
```

If the first fixture needs tail/control flow:

```text
ssavc4.make_flags
ssavc4.br
ssavc4.cond_br
```

For `saxpy_full`:

```text
ssavc4.tmu.request
ssavc4.tmu.read
```

For reductions:

```text
ssavc4.rotate
ssavc4.pack
ssavc4.unpack
```

For cooperative fixtures:

```text
ssavc4.sema.acquire
ssavc4.sema.release
ssavc4.barrier
ssavc4.vpm.read
ssavc4.vpm.write
```

### 17.2 Deferred operations

Defer until explicitly needed:

```text
ssavc4.dma.desc
ssavc4.dma.start
ssavc4.dma.wait
TMU texture/cubemap modes
TMU1
SFU beyond sfu_recip
host interrupt
physical QPU diagnostic reads
raw V3D scheduler controls
```

---

## 18. Fixture strategy

### 18.1 M2 fixtures remain scheduled-regression tests

Do not delete or replace M2 scheduled-input fixtures. They are the regression suite for the scheduled sink.

### 18.2 M3 fixtures are SSAVC4 input fixtures

Add M3 fixtures under a new tree such as:

```text
compiler/test/CodeGen/SSAVC4/Hardware/Run/<fixture>/input.mlir
compiler/test/CodeGen/SSAVC4/Hardware/Run/<fixture>/expected.json
compiler/test/CodeGen/SSAVC4/Hardware/Run/<fixture>/candidate/...
```

M3 fixtures may reuse M2 candidate harnesses and expected semantics when appropriate, but the input IR must be SSAVC4. The generated artifacts must still match the M2 artifact shape:

```text
manifest.json
layout.json
kernels/<public_name>.qasm
kernel_launch.c
kernel_launch.h
<code_symbol>.c/.h
```

### 18.3 Initial fixture order

Recommended order:

```text
1. minimal_thrend_ssavc4
2. vector_store_smoke_ssavc4 or global_store_coalesced_multi_ssavc4
3. global_store_coalesced_multi_ssavc4 with tail/branch support
4. saxpy_full_ssavc4
5. warp_reduce_sum_ssavc4 or warp_prefix_sum_ssavc4
6. qpu_barrier_syncthreads_ssavc4 or shared_transpose_16x16_ssavc4
```

Do not start with GEMV, matmul, attention, or full cooperative shared-memory kernels.

---

## 19. M3 milestone package

M3 should use the generic milestone automation.

Add:

```text
pro_scripts/milestones/vc4-ssavc4-m3.json
pro_scripts/vc4_ssavc4_m3_worklist.json
pro_scripts/vc4_ssavc4_m3_verifications.json
pro_scripts/vc4_ssavc4_m3_context_profiles.json
pro_scripts/prompts/vc4_ssavc4_m3/constitution.md
pro_scripts/prompts/vc4_ssavc4_m3/output_contract.md
pro_scripts/prompts/vc4_ssavc4_m3/slice_contract.md
pro_scripts/prompts/vc4_ssavc4_m3/slice*_handoff.md
```

The M3 descriptor must point to the M3 worklist/verifications/context profiles/prompts and its own state root, e.g.:

```json
{
  "schema_version": 1,
  "milestone": "vc4-ssavc4-m3",
  "title": "VC4 Milestone 3: SSAVC4 to scheduled VC4",
  "worklist": "pro_scripts/vc4_ssavc4_m3_worklist.json",
  "verifications": "pro_scripts/vc4_ssavc4_m3_verifications.json",
  "context_profiles": "pro_scripts/vc4_ssavc4_m3_context_profiles.json",
  "prompt_template_dir": "pro_scripts/prompts/vc4_ssavc4_m3",
  "state_root": ".vc4_auto/ssavc4_m3",
  "candidate_state_root": ".vc4_auto/ssavc4_m3",
  "default_from_slice": "m3-00-scaffold",
  "default_timeout_sec": 7200,
  "hardware_required_by_default": true
}
```

M2 and later milestones are cumulative. M3 verification must include explicit M2 scheduled-regression checks at least at final acceptance, and preferably at cumulative checkpoints.

---

## 20. Recommended M3 slices

### M3-00: milestone package and SSAVC4 scaffold

Deliver:

```text
M3 milestone descriptor
M3 worklist/verifications/context profiles/prompts
SSAVC4 dialect skeleton
CMake integration
vc4-opt registration
empty/minimal roundtrip tests
```

Verification:

```bash
python3 pro_scripts/vc4_milestone_verifier.py mechanisms --repo "$PWD"
ninja -C compiler/build vc4-opt
ninja -C compiler/build check-vc4
vc4-opt --show-dialects | grep ssavc4
```

### M3-01: type model and pure ops

Deliver:

```text
SSAVC4 value-type predicates
load_imm
element_number
splat
mov
alu.add
alu.mul
pack/unpack/rotate if straightforward
roundtrip/invalid tests
canonicalization for trivial mov if desired
```

No hardware required.

### M3-02: minimal lowering skeleton to scheduled VC4

Deliver:

```text
--convert-ssavc4-to-vc4
ssavc4.module/func -> vc4.module/func
metadata copy
thread_end -> scheduled thrend epilogue
load_imm / simple ALU -> scheduled vc4.qpu.*
```

Fixture:

```text
minimal_thrend_ssavc4 or simple constant/no-op kernel
```

Acceptance:

```bash
vc4-opt --convert-ssavc4-to-vc4 input.mlir | vc4-opt --vc4-verify-emit-contract ...
vc4-codegen generated scheduled output --emit-bundle <dir>
```

### M3-03: first observable global store

Deliver:

```text
ssavc4.vdw.store
minimal VPM/VDW store lowering
conservative mutex/wait sequence if needed
generate/assemble/build/run vector_store_smoke_ssavc4 or global_store_coalesced_multi_ssavc4
```

Acceptance includes hardware run and `expected.json`.

### M3-04: branch/tail lowering

Deliver:

```text
make_flags
br
cond_br
branch layout
3 nop delay slots
tail-control global store fixture
```

Acceptance:

```text
scheduled branch verifier passes
qasm assembles
hardware expected_json passes
```

### M3-05: TMU direct load and saxpy

Deliver:

```text
tmu.request/tmu.read direct tmu0 raw32
r4 lifetime preservation
saxpy_full_ssavc4
```

Acceptance:

```text
generate/assemble/build/run saxpy_full_ssavc4
M2 saxpy_full scheduled fixture still passes
```

### M3-06: reductions and rotate/dataflow pressure

Deliver:

```text
rotate lowering
more robust liveness
more robust register allocation
warp_reduce_sum_ssavc4 or warp_prefix_sum_ssavc4
```

### M3-07: cooperative barrier basics

Deliver:

```text
sema acquire/release
barrier op
cooperative resource metadata validation
qpu_barrier_syncthreads_ssavc4 or smaller barrier smoke
```

### M3-08: shared VPM cooperative fixture

Deliver:

```text
vpm read/write subset
shared_transpose_16x16_ssavc4 or reduced shared-VPM fixture
```

### M3-09: hardening and M2 regression

Deliver:

```text
no fixture-name special case checks
source contract checks
conversion invalid tests
full check-vc4
M2 scheduled fixture no-hardware smoke
targeted hardware runs for M3 fixtures
```

### M3-10: final M3 acceptance

Deliver:

```text
full M3 verifier pass
full M2 generic verifier pass or required M2 cumulative subset
final documentation
handoff for M4 vc4tile -> ssavc4
```

---

## 21. Verification rules

### 21.1 Always run after compiler-source changes

```bash
ninja -C compiler/build vc4-codegen vc4-opt
ninja -C compiler/build check-vc4
```

### 21.2 Scheduled output verification

Every SSAVC4 lowering test that produces scheduled `vc4` must run the existing scheduled checks where applicable:

```text
--vc4-verify-emit-contract
--vc4-verify-scheduled-hardware-rules
--vc4-verify-scheduled-adjacent-hazards
--vc4-verify-scheduled-io-spacing
--vc4-verify-scheduled-peripheral-accesses
```

### 21.3 Artifact verification

For hardware-capable fixtures:

```bash
bash compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh <fixture> generate
bash compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh <fixture> assemble
bash compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh <fixture> build
bash compiler/test/CodeGen/VC4/Support/run_candidate_codegen_test.sh <fixture> run
python3 compiler/test/CodeGen/VC4/Support/check_vc4_test_result.py \
  compiler/test/CodeGen/VC4/Hardware/Run/<fixture>/expected.json \
  .vc4_auto/<m3-state>/hardware/<fixture>/candidate_work/.vc4_candidate_run_attempt.log
```

M3 may need an SSAVC4-specific wrapper that first converts SSAVC4 input to scheduled VC4 before invoking the existing M2 support runner. The artifact shape must remain the same.

### 21.4 Regression verification

At M3 final acceptance, run M2 through the generic milestone verifier to prove scheduled backend persistence:

```bash
python3 pro_scripts/vc4_milestone_verifier.py verify \
  --repo "$PWD" \
  --milestone-config pro_scripts/milestones/vc4-codegen-m2.json \
  --slice m2-00-scaffold \
  --slice m2-01-manifest-v2 \
  --slice m2-02-program-bundle-assembly \
  --slice m2-03-persistent-program-layout \
  --slice m2-04-device-heap \
  --slice m2-05-cuda-like-launch-abi \
  --slice m2-06-multi-kernel-chain-hardware \
  --slice m2-07-independent-vector-scheduler \
  --slice m2-08-cooperative-block-scheduler \
  --slice m2-09-memory-subsystem-matrix \
  --slice m2-10-final-acceptance \
  --out .vc4_auto/ssavc4_m3/manual/m2-regression-after-m3.json \
  --timeout-sec 7200 \
  --keep-going
```

---

## 22. Anti-footguns and lessons from M2

### 22.1 Do not special-case fixtures

M2 repeatedly needed source-contract checks to prevent fixture/public-name special cases. M3 lowering must be generic. It must not branch on strings such as:

```text
saxpy_full
global_store_coalesced_multi
gemv_naive_tail
warp_reduce_sum
qpu_barrier_syncthreads
shared_transpose_16x16
```

Fixture names may appear only in tests, worklists, expected JSON, and artifact paths.

### 22.2 Do not confuse reference raw runtime with active runtime

Reference folders may contain old `mailbox.c`, raw SRQ code, or handwritten launchers. That is historical/reference ground truth. Active generated candidates use generated `kernel_launch.c/h` plus invariant libpi runtime.

### 22.3 Branch immediates are dangerous

M2 fixture debugging showed scheduled branch immediates were easy to get wrong. M3 lowering must compute branch layout from final scheduled slot order, not by guessing constants during local op lowering.

### 22.4 Cooperative classification matters

`block_reduce_sum` and `shared_transpose_16x16` are cooperative-block fixtures because they use shared VPM/semaphore/block residency semantics. `warp_reduce_sum` and `warp_prefix_sum` remain independent-vector fixtures because they use per-QPU/vector behavior without block-wide shared state. M3 metadata/lowering must preserve this distinction.

### 22.5 Physical QPU diagnostics are not compiler semantics

`vpm_setup_clobber` and physical-QPU reservation behavior are hardware characterization. They should not drive normal SSAVC4 semantics.

### 22.6 Incremental bring-up beats broad op-surface implementation

Do not implement all TMU/VPM/DMA/SFU/sync ops in one early slice. Define only what is needed, verify it, then expand.

### 22.7 M2 is cumulative; M1 is historical

M1 infrastructure was useful, but M2 superseded/deprecated many M1 end-state assumptions. M2 and later milestones are cumulative. M3 must preserve M2.

---

## 23. Acceptance criteria for M3

M3 is accepted only when all of the following are true:

1. `ssavc4` parses, prints, verifies, and round-trips.
2. Pure ops are marked pure.
3. Effectful hardware ops are not pure and carry effects/tokens.
4. `--convert-ssavc4-to-vc4` lowers supported SSAVC4 kernels to scheduled `vc4`.
5. Lowered scheduled output passes existing scheduled verifiers.
6. `vc4-codegen --emit-bundle` produces M2-compatible artifacts from lowered SSAVC4 kernels.
7. At least one SSAVC4 global-store fixture runs on hardware and passes expected JSON.
8. `saxpy_full_ssavc4` or equivalent TMU+ALU+VDW fixture runs on hardware and passes expected JSON.
9. At least one reduction or cooperative fixture is implemented or explicitly scoped as a later M3 extension with verifier support, depending on final M3 worklist.
10. No lowering or emitter code branches on fixture names.
11. No normal lowering path reads physical QPU number for logical identity.
12. Existing M2 scheduled fixtures still pass through the generic M2 verifier.
13. The M3 milestone package uses generic milestone automation, not milestone-specific driver scripts.

---

## 24. Deferred items

Out of M3 v1 unless a later M3 slice explicitly adds them:

```text
vc4tile -> ssavc4 lowering
more spill placement optimization
aggressive ADD/MUL bundling
useful branch delay-slot filling
TMU texture/cubemap paths
TMU1 fixtures
full SFU suite beyond basic reciprocal, if any
physical-QPU diagnostic lowering
relaxing conservative VPM setup serialization based on vpm_setup_clobber
matmul/attention/layernorm/softmax high-level kernels
```

---

## 25. Locked design decisions

1. Create a new `ssavc4` dialect.
2. Preserve scheduled `vc4` as the only artifact emitter sink.
3. Do not reintroduce structured `vc4`.
4. Use MLIR builtin scalar/vector types for ordinary data.
5. Add only minimal SSAVC4 custom types: token, descriptors as needed, flags.
6. Reuse only live scheduled-VC4 attrs/enums; define SSAVC4 attrs for descriptor concepts removed from `vc4`.
7. Reuse `"vc4.launch_abi"` and `"vc4.resource"` dictionaries.
8. Model pure ALU/value transforms as pure.
9. Model uniform/TMU/SFU/VPM/VDW/semaphore/mutex/barrier/thread-end as effectful.
10. Represent flags as restricted SSA pseudo-values.
11. Hide physical register addresses from SSAVC4 users.
12. Lower by selection, out-of-SSA, register allocation with spill-frame support, conservative scheduling, hazard insertion, and branch layout.
13. Start with conservative one-active-pipe scheduling and no useful delay-slot filling.
14. Use `ssavc4.vdw.store` as the first practical 1:N low-level global-store op.
15. Bring up hardware fixtures incrementally: store, branch/tail, TMU/saxpy, reductions, cooperative barriers.
16. Keep M3 below `vc4tile` lowering and above scheduled `vc4`; producer lowering into `vc4tile` is later work.
17. M3 final verification must prove both SSAVC4 progress and M2 scheduled-backend persistence.


## Register allocation and future spill-frame policy

M3 originally planned a conservative no-spill allocator, but the current SSAVC4 lower half now includes spilling support and block-argument/edge-copy lowering. Future milestones should treat spilling and SSAVC4 block arguments as existing lower-half capabilities, while still generating low-pressure IR where practical.

The lowering implementation remains structured around allocator and scheduler phases. The pipeline should separate instruction template selection, virtual value/liveness analysis, physical allocation and spill planning, scheduling, hazard insertion, and branch layout.

Spilling is not represented as public `ssavc4.push`, `ssavc4.pop`, or stack operations. Spill loads/stores are lowering-internal templates inserted by the allocator/spill planner. Upstream producer lowering to `vc4tile` must not know whether later SSAVC4 values are spilled.

The spill target is a private per-logical-request spill frame in global GPU memory, allocated from the existing VC4 program heap by generated launcher/runtime support. The compiler statically computes frame bytes per request from spill slots. Shared/VPM spilling is deferred as a later optimization because it consumes block-scoped on-chip resources and affects cooperative residency.
