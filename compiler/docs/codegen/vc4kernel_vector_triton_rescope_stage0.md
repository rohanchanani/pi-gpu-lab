# VC4Kernel / Vector / Triton Rescope Roadmap

**Status:** Stage 0 design document; no implementation patch.  
**Date:** 2026-05-30.  
**Input context snapshot:** `vc4kernel_rescope_stage0_context_20260530T002743Z.zip`, repo HEAD `fd244a005d306b7a6c12ddca1d6ea16909297eaf`, branch `compiler`.  
**Purpose:** Lock the post-M5 architectural rescope and give implementation agents a concrete roadmap from the current `vc4tile` state to v1 real-Triton lowering onto VC4 hardware.

---

## 1. Executive decision

We are revising the layer above SSAVC4.

The current repo contains a mature lower half:

```text
SSAVC4
  -> scheduled vc4
  -> vc4-codegen artifacts
  -> libpi-backed runtime
  -> VC4 hardware
```

It also contains a rich `vc4tile` dialect from M4/M5. The useful part is the **target-specific kernel core**: kernel wrappers, formal ABI args, program/block/warp/lane identity, masks, global load/store fragments, shared VPM/VDR/VDW movement, barriers, resource metadata, and core CFG legality. The part we are revising is the **ergonomic surface** that tried to act like a custom tile language with `tile_load`, `tile_store`, `copy_tile`, `tile_add`, `tile_reduce`, `tile_dot`, `tile_matmul`, `tile_contract`, and symbolic tile views.

The new architecture is:

```text
Triton-emitted TTIR / tt dialect
        ↓
standard MLIR value layer
  vector + memref + arith + math + scf/cf
        ↓
VC4 kernel planning dialect
  temporarily implemented by/rescoped from vc4tile core,
  eventually named vc4kernel
        ↓
SSAVC4
        ↓
scheduled vc4
        ↓
artifacts/runtime/hardware
```

The most important correction is this:

> Standard MLIR `vector`/`memref`/`arith`/`scf` should become the producer-facing value layer. The old custom `vc4tile` ergonomic surface should not be the primary lowering target for Triton or later IREE/Linalg. The target-specific dialect below `vector` should be `vc4kernel`, a rescope/rename of the current `vc4tile` core.

Implementation order must be bottom-up:

```text
1. Document the rescope.
2. Freeze/rescope current vc4tile core as vc4kernel.
3. Prove vc4kernel -> ssavc4 -> scheduled vc4 -> hardware.
4. Quarantine/deprecate the old vc4tile ergonomic surface as a producer target.
5. Add handwritten vector/memref/arith -> vc4kernel lowering.
6. Prove handwritten vector kernels on hardware.
7. Add real Triton-emitted TTIR ingestion.
8. Lower a strict TTIR v1 subset to vector/memref/arith.
9. Prove canonical Triton vector-add/SAXPY/tail kernels on hardware.
10. Extend later to reductions, dot/contract/matmul, richer layouts, and IREE.
```

The logical pipeline remains top-down, but the implementation must be bottom-up so each new layer has a hardware-verifiable path.

---

## 2. Why this rescope is necessary

### 2.1 What went wrong conceptually

M5 intentionally built an ergonomic `vc4tile` surface inspired by tile-kernel programming. That produced useful mechanisms, tests, and hardware evidence. However, the interpretation of that surface as the future producer-facing language is now the wrong long-term direction.

The old M5 model was roughly:

```text
Triton / IREE / producer IR
  -> vc4tile ergonomic surface
  -> canonicalize-vc4tile-surface
  -> plan-vc4tile-copies
  -> vc4tile core
  -> ssavc4
```

This made sense while we were thinking of VC4Tile as a CuTe/CuTile/ThunderKittens-like tile DSL. The corrected understanding is:

- CuTe is primarily layout/tensor algebra and compile-time mapping machinery, not a standalone MLIR tile language that should be cloned.
- NVIDIA/Triton-like tile programs already have a source IR (`tt`) and a GPU target-planning stack.
- MLIR already has a standard value layer for fixed-size vector/tile computation: `vector`, with `vector.transfer_read`, `vector.transfer_write`, `vector.contract`, `vector.reduction`, `vector.mask`, `vector.transpose`, `vector.step`, etc.
- IREE/Linalg naturally converges toward `linalg`/`tensor`/`memref`/`vector`, not toward a custom VC4 tile surface.

Therefore, the custom ergonomic `vc4tile` surface duplicates infrastructure MLIR already provides. The target-specific part of the current dialect should move down and become `vc4kernel`.

### 2.2 What was still right

The mistake was not building a target-specific layer above SSAVC4. That layer is still necessary.

VC4 has hardware facts that must be represented before SSAVC4:

```text
16 SIMD lanes per QPU warp
up to 12 physical QPU/warp slots
single global 4 KiB user-visible VPM shared-memory window
TMU global load path
VDR/VCD global-to-VPM load path
VPM QPU read/write path
VDW global store path
semaphore/barrier protocol
CUDA-like logical request/block/warp/lane metadata
raw u32 device pointers and uniform streams
```

Those are not generic MLIR `vector` facts. They belong in a VC4-specific kernel-planning dialect.

The corrected split is:

```text
vector/memref/arith/scf/cf:
  generic fixed-size value semantics and memory transfer semantics

vc4kernel:
  VC4 execution-plan semantics: fragments, VPM, VDR/VDW/TMU paths,
  barrier/resource/ABI planning, and predicates lowered to VC4-legal fragments

ssavc4:
  target machine SSA: uniforms, element_number, ALU ops, flags, branches,
  TMU/VDR/VPM/VDW/semaphore/barrier ops
```

---

## 3. Evidence from the current repo snapshot

### 3.1 The current `vc4tile` core/surface split already exists

The current `VC4TileOps.td` contains both target-kernel core operations and a large ergonomic surface. Core-like operations include:

```text
vc4tile.kernel
vc4tile.return
vc4tile.program_id
vc4tile.block_id
vc4tile.warp_id
vc4tile.lane_id
vc4tile.lane_range
vc4tile.thread_id
vc4tile.core_mask_all / core_tail_mask / core_tile_rect_mask / core_tile_bounds_mask
vc4tile.masked_load_global
vc4tile.masked_store_global
vc4tile.rotate
vc4tile.reduce
vc4tile.shared_alloc
vc4tile.shared_load
vc4tile.shared_store
vc4tile.shared_store_global
vc4tile.vdr_load_tile
vc4tile.barrier
```

Surface-like operations include:

```text
vc4tile.tile_descriptor
vc4tile.tile_load
vc4tile.tile_store
vc4tile.copy_tile
vc4tile.tile_view
vc4tile.tile_subview
vc4tile.transpose_view
vc4tile.shared_tile_alloc
vc4tile.tile_fill
vc4tile.tile_broadcast
vc4tile.tile_add
vc4tile.tile_sub
vc4tile.tile_mul
vc4tile.tile_select
vc4tile.tile_reduce
vc4tile.row_reduce
vc4tile.warp_reduce
vc4tile.block_reduce
vc4tile.tile_dot
vc4tile.tile_contract
vc4tile.tile_matmul
vc4tile.surface_placeholder
```

The pass implementation already recognizes this distinction. `isVC4TileSurfaceOp` explicitly classifies the surface ops and emits an ordering diagnostic requiring `--canonicalize-vc4tile-surface` and `--plan-vc4tile-copies` before core verification or conversion. The current pass pipeline contains:

```text
--canonicalize-vc4tile-surface
--plan-vc4tile-copies
--legalize-vc4tile-core-cfg
--verify-vc4tile-core
--convert-vc4tile-to-ssavc4
```

The `--verify-vc4tile-core` checker is already close to the future `--verify-vc4kernel` checker. It rejects raw `scf`, index-typed leakage, producer dialects such as `gpu`, `tt`, `ttg`, `nvgpu`, `iree`, `stablehlo`, `linalg`, `tensor`, `memref`, `spirv`, `nvvm`, and `rocdl`, and only allows a restricted set of VC4Tile, `arith`, `cf`, and `vector.splat` operations in lowering-ready core.

This means we do not need to invent `vc4kernel` from nothing. We need to rescope and rename the current core boundary, then remove the old assumption that the ergonomic surface is the future producer target.

### 3.2 The current lower half is already the right target stack

SSAVC4 exists as a target-specific machine-SSA dialect with operations such as:

```text
ssavc4.module
ssavc4.func
ssavc4.thread_end
ssavc4.load_imm
ssavc4.element_number
ssavc4.uniform.read
ssavc4.splat
ssavc4.mov
ssavc4.cond_select
ssavc4.alu.add
ssavc4.alu.mul
ssavc4.pack
ssavc4.unpack
ssavc4.rotate
ssavc4.make_flags
ssavc4.br
ssavc4.cond_br
ssavc4.sema.acquire
ssavc4.sema.release
ssavc4.barrier
ssavc4.tmu.request
ssavc4.tmu.read
ssavc4.vpm.write
ssavc4.vpm.read
ssavc4.vdr.load
ssavc4.vdw.store
ssavc4.vdw.store_vpm
```

The M3 design correctly established SSAVC4 as pre-register-allocation/pre-scheduling machine SSA, with scheduled `vc4` remaining the QASM-near artifact sink. The new rescope preserves that boundary. `vc4kernel` must not become an algorithmic sink with `matmul`-as-machine-instruction semantics. `vector.contract` and similar semantic operations must lower into fragment plans and eventually explicit SSAVC4 operations.

### 3.3 The runtime ABI favors logical memrefs in the value layer, raw pointers in launch ABI

The runtime exposes `vc4_deviceptr_t` as `uint32_t`, uses `vc4_launch_request_info` for logical request/block/warp/barrier/spill metadata, and launches with uniform-packing callbacks. It also exposes explicit device memory APIs: `vc4Malloc`, `vc4MemcpyHtoD`, `vc4MemcpyDtoH`, `vc4MemcpyDtoD`, and `vc4MemsetD8`.

Therefore the value-layer ABI should be:

```mlir
memref<?xi32, #vc4.global>
memref<?xf32, #vc4.global>
```

but the launch ABI lowering should be:

```text
one logical memref argument -> one raw u32 device-pointer uniform
```

No full host memref descriptor is introduced for VC4 v1.

### 3.4 M5 evidence is useful but partially superseded

The M5 design documents are now partly superseded. They locked useful decisions:

```text
32-bit-only executable semantics for M5 and first producer-lowering drafts
copy planning as a real target-specific concern
hardware as the gold standard
no direct VC4Tile -> scheduled VC4 shortcut
```

However, they also framed `tile_contract`, `tile_dot`, `tile_matmul`, `tile_reduce`, `tile_load`, `tile_store`, and `copy_tile` as ergonomic VC4Tile producer-facing contracts. That is the part being revised. The semantic need remains, but the producer-facing representation changes:

```text
old: producer -> vc4tile.tile_contract / tile_load / tile_reduce / ...
new: producer -> vector.contract / vector.transfer_read / vector.reduction / ...
     then vector -> vc4kernel plans VC4 fragments/resources
```

### 3.5 Automation lessons must shape the next milestone

The mistakes ledger records repeated issues that must be accounted for in the new roadmap:

```text
mechanical repair prompts must be milestone-specific
Codex mechanical budget must remain at least three attempts
verifier/spec mismatches must be fixed in verifier/spec, not by output hacks
source-native ODS op definitions must not be verified by comment-gamable raw text scans
feature gates must match real feature ownership
candidate/hardware tests must use fresh candidate generation and device-derived result checks
cumulative prefix validation must happen before commit
```

The rescope milestone should be conservative about feature ownership: do not mark a feature implemented until dialect, invalid-diagnostic, lowered-IR, artifact, and hardware-reference evidence exist where applicable.

---

## 4. External MLIR/Triton evidence

### 4.1 MLIR `vector` is the right generic value layer

The MLIR Vector dialect documentation states that MLIR supports multi-dimensional vector types and custom operations on SSA vector values, and that generic retargetable higher-order vector types carry semantic information useful for transformations. It also says the vector abstractions separate concerns between operations on `memref` buffers and operations on vector values.

Important vector operations for this compiler are:

```text
vector.transfer_read
vector.transfer_write
vector.contract
vector.reduction
vector.mask
vector.create_mask
vector.transpose
vector.step
vector.broadcast / shape_cast / extract / insert
```

The Vector docs explicitly describe `vector.transfer_read` as a mid-level abstraction whose “super-vector” granularity is generally not representable with a single hardware register. That is exactly the situation on VC4: `vector<4x4xi32>` or `vector<8x8xf32>` may decompose into multiple 16-lane fragments, VPM rows, or transfer fragments.

`vector.contract` also directly models dot/matmul-style contractions with affine indexing maps and reduction iterator types. That is a better generic producer-facing representation than custom `vc4tile.tile_contract`.

### 4.2 MLIR `nvgpu` is the closest layering precedent, not a template to copy

The MLIR NVGPU dialect is described as a bridge between higher-level target-agnostic GPU/Vector dialects and the lower-level target-specific NVVM dialect for NVIDIA GPUs. It represents PTX-specific operations while still using `memref` and `vector` for memory and target-specific register operands.

That is the right analogy:

```text
MLIR NVIDIA-like layering:
  gpu/vector/memref
    -> nvgpu
    -> nvvm/llvm/ptx

VC4 layering:
  vector/memref/arith/scf/cf
    -> vc4kernel
    -> ssavc4
    -> scheduled vc4/qasm artifacts
```

`vc4kernel` should play a role analogous to a VC4-specific planning dialect. It should not replace `vector`.

### 4.3 Triton TTIR should be the producer input, not TTGIR/NVIDIA-specific IR

The Triton `tt` dialect is Triton IR in MLIR and depends on Arith, Math, SCF, and CF. Triton ops include exactly the producer features required for v1 and beyond:

```text
tt.func
tt.get_program_id
tt.load
tt.store
tt.dot
tt.broadcast
tt.splat
tt.reduce / scan-style reductions
arith/math/scf/cf dependencies
```

For M6 v1, consume TTIR emitted by the real Triton compiler. Do not target CUDA TTGIR, NVIDIA-specific `ttnvgpu`, `nvgpu`, `nvvm`, PTX, or cubin paths.

---

## 5. Final layer definitions

### 5.1 Standard MLIR value layer

The value layer is the producer-facing IR that future Triton and IREE/Linalg lowerings should target.

Allowed v1 dialects:

```text
arith
math, only where explicitly supported later
scf/cf
memref, with VC4 global memory-space convention
vector
func or vc4kernel.kernel wrapper, depending on staging
```

V1 value-layer operations:

```text
vector.step
vector.create_mask / vector.constant_mask / vector.mask
vector.transfer_read
vector.transfer_write
vector.contract, future after elementwise v1
vector.reduction, future after elementwise v1
vector.transpose / vector.broadcast / vector.extract / vector.insert as needed
arith.addi/subi/muli, addf/subf/mulf
arith.cmpi/cmpf
arith.select
scf.for/scf.if where already legalizable
cf.br/cf.cond_br after legalization
```

This layer answers:

> What is the fixed-size computation and generic memory-transfer semantics?

It does **not** answer:

```text
Which VC4 memory path is used?
Which VPM rows are allocated?
Which VDW/VDR setup form is emitted?
Which semaphores implement a barrier?
Which launch uniform index holds logical_request?
```

### 5.2 VC4 kernel planning layer (`vc4kernel`)

The VC4 kernel layer answers:

> How is this fixed-size computation represented and executed on VC4 hardware?

Core concepts:

```text
kernel wrapper and symbol/public name
formal ABI args and raw uniform assignments
program_id / block_id / warp_id / lane_id / lane_range / thread_id
16-lane row fragments
shared VPM allocation / read / write
VDR/VCD global-to-VPM load plans
VDW VPM-to-global store plans
TMU/register global load plans
masked global store plans
normalized predicates and fragment plans
rotate/reduce fragment ops
barriers and semaphore resource plans
resource metadata: schedule mode, warps per block, shared VPM rows/bytes, semaphores, full residency
core CFG using cf.br/cf.cond_br/block args
```

This layer is currently implemented by `vc4tile` core. Stage 1 should make that boundary explicit as `vc4kernel`, even if the first implementation uses aliases or compatibility names.

### 5.3 SSAVC4

SSAVC4 answers:

> What virtual machine-SSA operations implement the VC4 kernel plan?

It contains:

```text
uniform reads
lane element_number
SSA ALU dataflow
flags and conditional branches
successor operands / block args
TMU requests and reads
VPM reads/writes
VDR loads
VDW stores
semaphores/barriers
thread_end
launch/resource metadata
```

SSAVC4 must not contain high-level algorithmic operations like `matmul` or generic `vector.transfer_read`. If those survive into SSAVC4, the boundary failed.

### 5.4 Scheduled VC4

Scheduled `vc4` remains the QASM-near sink:

```text
vc4.module
vc4.func
vc4.qpu.ldi
vc4.qpu.bundle
vc4.qpu.sema
vc4.qpu.branch
scheduled peripheral/VPM/TMU/VDW pseudoops where already modeled
```

No producer or `vc4kernel` pass may bypass SSAVC4 and lower directly to scheduled VC4.

---

## 6. Salvage / quarantine map for current `vc4tile`

### 6.1 Preserve as `vc4kernel` core

Preserve or rename these semantics:

```text
vc4tile.kernel              -> vc4kernel.kernel
vc4tile.return              -> vc4kernel.return
vc4tile.program_id          -> vc4kernel.program_id
vc4tile.block_id            -> vc4kernel.block_id
vc4tile.warp_id             -> vc4kernel.warp_id
vc4tile.lane_id             -> vc4kernel.lane_id
vc4tile.lane_range          -> vc4kernel.lane_range
vc4tile.thread_id           -> vc4kernel.thread_id
vc4tile.core_mask_*         -> vc4kernel.predicate / predicate fragments
vc4tile.masked_load_global  -> vc4kernel.global_load_fragment, v1 compatibility ok
vc4tile.masked_store_global -> vc4kernel.global_store_fragment, v1 compatibility ok
vc4tile.rotate              -> vc4kernel.fragment_rotate
vc4tile.reduce              -> vc4kernel.fragment_reduce
vc4tile.shared_alloc        -> vc4kernel.vpm_alloc / shared_alloc
vc4tile.shared_load         -> vc4kernel.shared_load_fragment
vc4tile.shared_store        -> vc4kernel.shared_store_fragment
vc4tile.shared_store_global -> vc4kernel.shared_to_global_fragment
vc4tile.vdr_load_tile       -> vc4kernel.global_to_vpm_fragment
vc4tile.barrier             -> vc4kernel.barrier
```

The exact op names do not need to change in one giant patch. The important Stage 1 outcome is semantic: define the core boundary as `vc4kernel` and prevent future producer work from targeting the old surface.

### 6.2 Deprecate as producer-facing surface

Do not use these as the future Triton/IREE target:

```text
vc4tile.tile_load
vc4tile.tile_store
vc4tile.copy_tile
vc4tile.tile_view
vc4tile.tile_subview
vc4tile.transpose_view
vc4tile.tile_fill
vc4tile.tile_broadcast
vc4tile.tile_add/sub/mul/select
vc4tile.tile_reduce / row_reduce / warp_reduce / block_reduce
vc4tile.tile_dot
vc4tile.tile_contract
vc4tile.tile_matmul
```

Do not delete them immediately. Keep them temporarily for:

```text
M5 regression tests
hardware fixtures already built around them
comparison while implementing vector -> vc4kernel
migration aids
```

But do not build M6 around them.

### 6.3 Reinterpret as implementation assets

These M5 mechanisms are still valuable:

```text
copy planning code
predicate normalization/fragments
shared VPM/VDR/VDW lowering code
SCF legalization
core verification
candidate hardware runner
hardware fixture matrix
precision diagnostics
```

Move their ownership conceptually:

```text
old ownership:
  vc4tile surface -> canonicalize/plan -> vc4tile core

new ownership:
  vector/memref value layer -> vector-to-vc4kernel planning -> vc4kernel core
```

---

## 7. Logical pipeline and implementation order

### 7.1 Logical pipeline

The eventual v1 Triton pipeline is:

```text
Triton source
  -> real Triton compiler emits TTIR / tt dialect MLIR
  -> TTIR-to-standard-value lowering
  -> vector/memref/arith/scf/cf value layer
  -> vector-to-vc4kernel planning
  -> vc4kernel core CFG
  -> vc4kernel-to-ssavc4
  -> ssavc4-to-scheduled-vc4
  -> vc4-codegen artifact bundle
  -> libpi-backed VC4 hardware execution
```

### 7.2 Implementation order

Implementation must be bottom-up:

```text
Stage 0: Design document and rescope lock.
Stage 1: Rescope/freeze vc4tile core as vc4kernel.
Stage 2: Prove vc4kernel -> ssavc4 -> hardware.
Stage 3: Quarantine old ergonomic surface as non-producer-target compatibility.
Stage 4: Implement handwritten vector/memref/arith -> vc4kernel.
Stage 5: Prove handwritten vector kernels on hardware.
Stage 6: Add real Triton TTIR emission/ingestion infrastructure.
Stage 7: Implement strict TTIR v1 -> vector lowering.
Stage 8: Prove real Triton elementwise kernels on hardware.
Stage 9+: Extend to reductions, dot/contract/matmul, richer masks/layouts, IREE.
```

The bottom-up order is mandatory because the compiler must be hardware-verifiable at every new boundary. A top-down attempt would mix TTIR parsing, vector semantics, VC4 fragmentation, launch ABI, SSAVC4 lowering, and hardware debugging all at once.

---

## 8. Stage-by-stage roadmap

## Stage 0: freeze and document the rescope

### Goal

Create this document and establish it as the authoritative planning artifact for the post-M5/pre-M6 transition.

### Deliverables

```text
compiler/docs/codegen/vc4kernel-vector-triton-rescope-stage0.md
```

### Decisions locked

```text
old vc4tile ergonomic surface is not the future producer target
standard MLIR vector/memref/arith/scf/cf is the future value layer
current vc4tile core is the source of vc4kernel
implementation order is bottom-up
Triton v1 consumes real emitted TTIR, not handwritten fake TTIR and not TTGIR
logical memrefs lower to raw u32 device-pointer uniforms
```

### Non-goals

```text
no compiler implementation change
no pass rename yet
no Triton dependency yet
no deletion of M5 surface ops
```

---

## Stage 1: rescope/freeze current `vc4tile` core as `vc4kernel`

### Goal

Make the semantic boundary real without disrupting the lower half.

### Recommended implementation strategy

Do **not** start with a giant textual rename. Start with compatibility naming and pass aliases.

Add/prepare conceptual pass names:

```text
--verify-vc4kernel
--convert-vc4kernel-to-ssavc4
```

These may initially call the existing implementations:

```text
--verify-vc4tile-core
--convert-vc4tile-to-ssavc4
```

Add a design/diagnostic note that `vc4tile` core is the legacy spelling of `vc4kernel` during migration.

### Required source changes

Likely touched areas:

```text
compiler/include/vc4/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.h
compiler/lib/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.cpp
compiler/tools/vc4-opt/vc4-opt.cpp
compiler/docs/codegen/...
compiler/test/Conversion/VC4TileToSSAVC4/...
compiler/test/Dialect/VC4Tile/...
```

Do not alter SSAVC4 or scheduled VC4 semantics in this stage.

### Required verification

```text
ninja -C compiler/build vc4-opt
ninja -C compiler/build check-vc4
vc4-opt --help contains verify-vc4kernel / convert-vc4kernel-to-ssavc4 aliases
existing vc4tile core fixtures still lower and run
new alias fixtures prove vc4kernel-named path is not a direct vc4 shortcut
implementation-integrity scan confirms no vector/triton producer lowering yet
```

### Acceptance criterion

A handwritten core kernel can lower through:

```text
vc4kernel/legacy-vc4tile-core input
  -> ssavc4
  -> scheduled vc4
  -> artifact bundle
  -> hardware
```

The old surface is still present but not framed as the producer target.

---

## Stage 2: prove `vc4kernel -> ssavc4 -> hardware`

### Goal

Before adding any vector lowering, prove the new kernel layer is a stable hardware-proven target.

### Required fixture families

Adapt or preserve existing core fixtures:

```text
minimal thread end
formal args / launch ABI
program_id writeback
lane_range vector store
masked global store
masked global load + saxpy
SCF/core CFG loops and ifs
rotate/reduce
shared VPM roundtrip
VDR global-to-VPM load
VDW VPM-to-global store
barrier / cooperative block
predicate fragments needed for current tests
```

### Required verification

Every executable feature needs:

```text
dialect/roundtrip evidence
invalid diagnostic evidence
kernel-to-ssavc4 lowered-IR evidence
ssavc4-to-scheduled-vc4 artifact evidence
fresh candidate hardware run
CPU-reference/device-copyback comparison
implementation integrity scan
```

### Anti-shortcut rules

```text
no fixture-name special casing
no direct vc4kernel/vc4tile -> scheduled vc4
no host-side computation replacing device results
no fixed VC4_TEST_RESULT status=PASS
no stale .vc4_auto candidate reuse in required gates
no synthetic launch ABI builtins for kernels that do not use them
```

---

## Stage 3: quarantine/deprecate the old ergonomic surface

### Goal

Stop treating old M5 surface ops as the future frontend target.

### Required changes

- Add documentation that `vc4tile.tile_*` ops are legacy/compatibility surface ops.
- Ensure new milestones do not ask Triton lowering to emit them.
- Keep M5 regression tests runnable for now.
- Remove claims in docs/prompts that future Triton/IREE lowerings should target ergonomic VC4Tile surface ops.
- Add warnings or diagnostics only if they do not break existing committed tests; otherwise document deprecation first and migrate tests later.

### Do not do yet

```text
do not delete tile_load/tile_store/copy_tile/tile_contract/etc. immediately
do not break M5 regression fixtures
do not rewrite all current tests in one patch
```

---

## Stage 4: implement handwritten `vector/memref/arith -> vc4kernel`

### Goal

Prove the standard MLIR value layer can target VC4 hardware through `vc4kernel`.

### V1 value-layer ABI

Use:

```mlir
memref<?xi32, #vc4.global>
memref<?xf32, #vc4.global>
```

Lowering convention:

```text
memref argument -> one raw u32 device pointer uniform
indexing in vector/memref layer is element-based
byte offsets are introduced only when planning VC4 memory ops
```

### First supported value-layer subset

```text
vector<16xi32>
vector<16xf32>
vector.step
vector.transfer_read with contiguous 1D access, mask, zero padding
vector.transfer_write with contiguous 1D access and mask
arith.addi/subi/muli on vector<16xi32>
arith.addf/subf/mulf on vector<16xf32>
arith.cmpi/cmpf for simple tail masks
arith.select for vector masks where necessary
scalar i32/f32 constants and broadcasts
straight-line kernels first
```

### Example target transformation

Input value layer:

```mlir
%lanes = vector.step : vector<16xi32>
%offs = arith.addi %base_splat, %lanes : vector<16xi32>
%mask = arith.cmpi ult, %offs, %n_splat : vector<16xi32>
%xv = vector.transfer_read %x[%base], %zero, %mask
  : memref<?xi32, #vc4.global>, vector<16xi32>
%yv = vector.transfer_read %y[%base], %zero, %mask
  : memref<?xi32, #vc4.global>, vector<16xi32>
%zv = arith.addi %xv, %yv : vector<16xi32>
vector.transfer_write %zv, %z[%base], %mask
  : vector<16xi32>, memref<?xi32, #vc4.global>
```

Planned kernel layer:

```mlir
%lanes = vc4kernel.lane_range : vector<16xi32>
%pred = vc4kernel.predicate.row_tail ...
%x = vc4kernel.global_load_fragment ... {inactive = zero_fill}
%y = vc4kernel.global_load_fragment ... {inactive = zero_fill}
%z = arith.addi or vc4kernel.fragment_add ...
vc4kernel.global_store_fragment ... {inactive = preserve_destination}
```

### Required hardware tests

```text
handwritten_vector_add_i32
handwritten_saxpy_f32
handwritten_tail_n_0
handwritten_tail_n_1
handwritten_tail_n_17
handwritten_multi_request
```

These should not involve Triton yet.

---

## Stage 5: prove handwritten vector kernels on hardware

### Goal

Make the standard value-layer path hardware-proven before adding Triton.

### Required acceptance

For each fixture:

```text
source vector/memref/arith IR
  -> vector-to-vc4kernel
  -> verify-vc4kernel
  -> vc4kernel-to-ssavc4
  -> ssavc4-to-vc4
  -> vc4-codegen bundle
  -> hardware run
  -> CPU reference check
```

### Required integrity gates

```text
no TTIR in this stage
no old tile_* surface emission
no direct vector -> ssavc4 shortcut unless explicitly marked internal experimental and not accepted
no direct vector -> scheduled vc4
no fixture-name special casing
```

---

## Stage 6: add real Triton TTIR emission/ingestion infrastructure

### Goal

Use a real Triton compiler to emit canonical TTIR, then consume that TTIR in our compiler.

### Recommended dependency model

For v1:

```text
Use an external/source-built or pip-installed Triton environment to emit TTIR.
Check in emitted textual TTIR test artifacts plus original Triton Python source.
Add a repo script to regenerate TTIR from pinned Triton version/commit.
Do not make normal check-vc4 depend on regenerating TTIR.
```

For consumption:

```text
Build a separate Triton-aware bridge tool or optional vc4-opt mode that links/registers Triton dialects.
Do not parse TTIR text with ad hoc regexes.
Do not vendor a fake subset of TTIR syntax.
Do not consume TTGIR/NVIDIA-specific dialects for VC4 v1.
```

Recommended tool shape:

```text
vc4-triton-import
  input: emitted .ttir.mlir
  output: standard value-layer MLIR or directly vector->vc4kernel pipeline input
```

Alternative:

```text
vc4-opt --register-triton-dialects --convert-ttir-to-vc4-vector
```

But keep Triton registration optional so the core VC4 compiler still builds without Triton.

### Required repo artifacts per Triton fixture

```text
source.py
emitted.ttir.mlir
normalized-vector.mlir or imported.mlir, when useful for debugging
expected.json
candidate harness
```

---

## Stage 7: implement strict TTIR v1 -> vector lowering

### V1 supported TTIR subset

Support only:

```text
tt.func
tt.return
tt.get_program_id axis = 0
tt.arange with block size 16
tt.load with contiguous pointer tensor, mask, other = 0
tt.store with contiguous pointer tensor and mask
tt.broadcast / tt.splat patterns needed by elementwise kernels
arith.addi/addf/muli/mulf/subi/subf
arith.cmpi/cmpf
simple scalar constants
straight-line kernels; scf only if already verified through vector layer
```

Reject with deterministic diagnostics:

```text
TTGIR input
NVIDIA/AMD-specific dialects
program_id axis 1/2
block sizes other than 16
tt.dot, initially until vector.contract path is ready
tt.reduce, initially until vector.reduction path is ready
arbitrary gather/scatter
pointer expressions not affine base + arange + scalar offset
fp16/bf16/fp8/int8/int4/sub-32 types
tensor descriptors / TMA-like forms
generic sparse masks not handled by the predicate planner
```

### Lowering map

```text
tt.get_program_id(axis=0)
  -> vc4kernel.program_id or value-layer placeholder later consumed as vc4kernel.program_id

tt.arange(0, 16)
  -> vector.step : vector<16xi32>

tt.load(ptr + offsets, mask, other=0)
  -> vector.transfer_read from logical memref with mask and zero padding

tt.store(ptr + offsets, value, mask)
  -> vector.transfer_write with mask

tt.broadcast / tt.splat
  -> vector.broadcast / arith constants

tt.dot
  -> reject in v1a; later vector.contract

tt.reduce
  -> reject in v1a; later vector.reduction
```

### First Triton source fixture

```python
@triton.jit
def add_kernel(x, y, z, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    xv = tl.load(x + offs, mask=mask, other=0)
    yv = tl.load(y + offs, mask=mask, other=0)
    tl.store(z + offs, xv + yv, mask=mask)
```

Lock `BLOCK = 16` for v1.

---

## Stage 8: v1 Triton hardware acceptance

### Goal

Run canonical Triton-emitted TTIR through the full VC4 stack to hardware.

### Required fixtures

```text
triton_vector_add_i32_block16
triton_saxpy_f32_block16
triton_tail_n_0
triton_tail_n_1
triton_tail_n_17
triton_tail_n_31
triton_multi_request_large_n
unsupported_axis1_diagnostic
unsupported_fp16_diagnostic
unsupported_dot_before_contract_stage_diagnostic
```

### Acceptance path

```text
source.py
  -> real Triton compiler emits TTIR
  -> checked-in emitted.ttir.mlir consumed by vc4-triton-import or optional vc4-opt mode
  -> standard vector/memref/arith layer
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> vc4-codegen
  -> hardware
  -> CPU oracle
```

### Anti-shortcut checks

```text
source.py must exist beside emitted.ttir.mlir
emitted TTIR must include real tt.* operations
no handwritten fake TTIR-only acceptance
no direct TTIR -> ssavc4 shortcut
no direct TTIR -> scheduled vc4 shortcut
no old vc4tile.tile_* surface emission in accepted v1 path
no host-side oracle copied into device output
```

---

## Stage 9+: extensions after v1

After real Triton vector-add/SAXPY hardware proof:

```text
1. Add vector.reduction -> vc4kernel fragment_reduce.
2. Add tt.reduce -> vector.reduction.
3. Add vector.contract -> vc4kernel fragment_contract.
4. Add tt.dot -> vector.contract for 32-bit-only supported shapes.
5. Add 4x4/8x8/16x16 decomposition and VPM staging.
6. Add richer predicate normalization and sparse fallback diagnostics.
7. Add IREE/Linalg path via vector/memref value layer.
8. Add sub-32 precision/storage/packing after 32-bit producer paths are proven.
```

---

## 9. Verification architecture for the rescope milestone

Each feature should be verified at the right layer. Do not let a feature pass based only on text scans or non-executable roundtrips.

### 9.1 Required verification layers

For executable compiler features:

```text
source_products
build vc4-opt
build vc4-codegen
build check-vc4
dialect_contract or value-layer parse/roundtrip contract
invalid_diagnostic_contract
lowered_ir_contract
scheduled_artifact_contract
hardware_cpu_reference_contract
implementation_integrity_contract
final build-check_vc4-post
```

### 9.2 Hardware policy

Hardware is required for:

```text
new vc4kernel executable ops
new vector-to-vc4kernel executable lowering
new TTIR-to-vector executable lowering
new memory path / predicate path / barrier path
```

Hardware is not required for:

```text
pure documentation
metadata-only declarations
unsupported-diagnostic tests
external TTIR emission script smoke, if it does not touch codegen
```

### 9.3 Power-cycle policy

All required hardware tests should power-cycle first where the existing infrastructure supports it. This should be encoded in support runners or verifier environment policy, not left as a human reminder.

### 9.4 Integrity checks

Keep and extend these checks:

```text
no direct vc4kernel/vc4tile -> vc4
no direct TTIR -> ssavc4 accepted path unless explicitly nonacceptance experimental
no producer lowering in stages before Triton stage
no sub-32 executable lowering in v1
no verifier-gaming comments/literals
no fixed PASS hardware scripts
no stale candidate reuse in required gates
no fixture-name special casing
no host computation replacing device output
```

---

## 10. Concrete current-state audit summary

From the Stage 0 context snapshot:

```text
Current HEAD: fd244a005d306b7a6c12ddca1d6ea16909297eaf
Branch: compiler
Context files copied: 823
Current diff in collection: none recorded in git-status-short output
```

High-value current assets:

```text
compiler/include/vc4/Dialect/VC4Tile/IR/VC4TileOps.td
compiler/lib/Dialect/VC4Tile/IR/VC4TileOps.cpp
compiler/lib/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.cpp
compiler/include/vc4/Dialect/SSAVC4/IR/SSAVC4Ops.td
compiler/lib/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.cpp
compiler/test/CodeGen/VC4Tile/Hardware/Run/**
compiler/test/Conversion/VC4TileToSSAVC4/**
compiler/docs/codegen/vc4_cuda_mapping_guide.md
compiler/docs/codegen/vc4tile-m5-full-design.md
compiler/docs/codegen/vc4tile_cute_predication_semantics.md
pro_scripts/MISTAKES.md
```

Current source has already moved beyond simple M4/M5:

```text
semantic predicate carriers
large M5 hardware fixture corpus
CuTe-style predication hardening work
VDR/VCD/VDW support
SCF-for/if semantics with zero-trip and non-divisible loops
copy planning and predicate fragment planning
```

This strengthens the case for a rescope rather than a restart: many mechanisms are useful, but the producer-facing surface target should change.

---

## 11. Non-negotiable design policies

### 11.1 No direct shortcuts

Forbidden:

```text
Triton TTIR -> scheduled vc4
Triton TTIR -> ssavc4, as the accepted v1 path
vector -> scheduled vc4
vc4kernel/vc4tile -> scheduled vc4
producer -> old tile_* surface as the primary v1 path
```

Allowed only as internal diagnostics/prototypes, not acceptance evidence:

```text
temporary debug-only direct dumps
one-off translation scripts not used in required tests
```

### 11.2 Preserve ABI categories

```text
user/caller values:
  kernel formal args -> vc4.launch_abi.args[] -> uniform words

runtime builtins:
  program_id/logical_request, block_id, warp_id, warps_per_block, barrier metadata
  -> vc4.launch_abi.builtins[] only when used

lane identity:
  ssavc4.element_number-derived, not a uniform or launch builtin
```

Empty/no-formal kernels with no used runtime builtins must preserve:

```text
args=[]
builtins=[]
uniform_words_per_qpu=0
no ssavc4.uniform.read
```

### 11.3 32-bit-only producer v1

V1 supports only:

```text
i32/u32/f32 scalars
vector<16xi32>
vector<16xf32>
32-bit global/shared/register transfers
```

Reject:

```text
f16
bf16
fp8/fp4
int8/uint8
int4/uint4
packed sub-32 executable movement/computation
quantization/dequantization paths
```

Forward-looking metadata can remain in lower/core planning where useful, but no non-32-bit executable behavior may be accepted.

### 11.4 Use real TTIR

Triton acceptance requires:

```text
real Triton source
real Triton-emitted TTIR
checked-in emitted textual TTIR or deterministic regeneration script
Triton dialect-aware parsing/lowering
hardware output from device execution
```

Not accepted:

```text
handwritten fake tt.* input as sole evidence
regex parsing of TTIR
TTGIR/NVIDIA-specific lowering as VC4 input
```

---

## 12. What implementation agents should do next

The next implementation prompt after this document should be Stage 1, not vector lowering and not Triton lowering.

Stage 1 should ask for:

```text
1. Add/alias vc4kernel pass boundary names without breaking existing vc4tile tests.
2. Document legacy vc4tile-core spelling as compatibility.
3. Add tests proving the alias boundary still lowers through ssavc4.
4. Add implementation-integrity checks preventing direct vc4kernel/vc4tile -> vc4.
5. Keep all current M5/M4/M3 lower-half tests green.
6. Run hardware only for existing core fixtures affected by the pass boundary.
```

Stage 1 should not ask for:

```text
vector.transfer_read lowering
memref ABI changes
Triton dependency setup
old surface deletion
broad rename of every source file
```

---

## 13. Summary for future milestone package generation

A future milestone package should be generated around these slices:

```text
r0-00-rescope-doc-and-package
r0-01-vc4kernel-boundary-aliases
r0-02-vc4kernel-core-hardware-regression
r0-03-surface-quarantine-and-doc-cleanup
r0-04-logical-memref-abi-contract
r0-05-vector-transfer-arith-to-vc4kernel
r0-06-vector-hardware-acceptance
r0-07-triton-ttir-emission-corpus
r0-08-triton-dialect-aware-import-tool
r0-09-ttir-elementwise-to-vector
r0-10-real-triton-vector-add-hardware
r0-11-real-triton-saxpy-tail-hardware
r0-12-final-acceptance
```

This sequence intentionally gets to hardware before each new higher layer is trusted.

---

## 14. Source references used

Repo/context sources:

```text
vc4kernel_rescope_stage0_context_20260530T002743Z.zip
compiler/include/vc4/Dialect/VC4Tile/IR/VC4TileOps.td
compiler/lib/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.cpp
compiler/include/vc4/Dialect/SSAVC4/IR/SSAVC4Ops.td
compiler/lib/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.cpp
compiler/docs/codegen/vc4_cuda_mapping_guide.md
compiler/docs/codegen/vc4-ssavc4-m3-plan.md
compiler/docs/codegen/vc4tile-m5-full-design.md
compiler/docs/codegen/vc4tile-m5-compute-primitives-design.md
compiler/docs/codegen/vc4tile-m5-precision-roadmap.md
compiler/docs/codegen/vc4tile_cute_predication_semantics.md
pro_scripts/MISTAKES.md
external/libpi/include/vc4_runtime.h, as captured in uploaded context/search snippets
```

External design references:

```text
MLIR Vector dialect documentation
MLIR NVGPU dialect documentation
Triton tt dialect and TritonOps documentation
Triton NVIDIA backend compiler.py, used as architecture analogy only
```

---

## 15. Locked one-paragraph version

The post-M5 architecture should treat standard MLIR `vector`/`memref`/`arith`/`scf`/`cf` as the producer-facing value layer, not the old custom ergonomic `vc4tile.tile_*` surface. Current `vc4tile` core should be rescaled/renamed into `vc4kernel`, a VC4-specific kernel planning dialect that owns fragments, predicates, VPM/TMU/VDR/VDW movement, barriers, launch ABI, and resources. The bottom-up bring-up path is `vc4kernel -> ssavc4 -> scheduled vc4 -> hardware`, then `vector -> vc4kernel`, then real Triton-emitted `TTIR -> vector`. V1 Triton is 32-bit-only, consumes real emitted TTIR, supports a strict elementwise/tail subset first, and proves correctness through fresh VC4 hardware candidate execution and CPU-reference comparison.
