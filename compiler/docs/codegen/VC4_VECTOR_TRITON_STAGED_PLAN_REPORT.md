# VC4 Vector/Triton Staged Plan Report

**Project:** `rohanchanani/pi-gpu-lab`, branch `compiler`  
**Stage:** planning document for post-P13 `vector/Triton -> vc4kernel` work  
**Status:** design/specification artifact; not an implementation prompt package  
**Generated:** 2026-06-07  
**Primary audience:** future ChatGPT/Codex implementation chats that will generate phase prompt packages after the post-P13 context refresh.

---

## Current Post-Phase-7.5 Note

This report contains historical planning text that predates the locked C++
frontend path. Current accepted TTIR-to-VC4Value semantic lowering is:

```text
vc4-triton-opt --convert-triton-to-vc4-value
```

Historical references below to `vc4-triton-import` as a TTIR-to-value semantic
lowering tool are retired. Python tooling remains allowed for Triton
source-to-TTIR generation, snapshot regeneration/comparison, and TTIR inventory
only. `READY_FOR_TRITON` remains `NO`.

## Current Phase-8.25 Vertical Workflow Note

The standing workflow is now explicitly vertical. The project should not add
many value-layer capabilities before returning to TTIR. Each coherent
source-visible value capability or legality category should be followed by a
matching TTIR bridge/support/reject lock before the next major value capability,
unless the value phase only refactors internal lowering without adding a
source-visible capability or legality category.

Phase 8 value-layer control flow therefore leads to Phase 8.5 TTIR
control-flow bridge/support/reject lock, not directly to Phase 9. Phase 9 mask
classifier and richer memory legality leads to Phase 9.5 TTIR mask/memory
bridge/support/reject lock, not directly to Phase 10. `READY_FOR_TRITON`
remains `NO`.

## 0. Purpose

This report records the current state of the Vector/Triton plan after the latest design discussion. It updates and extends the earlier pre-P9 Vector/Triton design-decision report with the decisions now made about `vc4value`, program IDs, memref ABI, Triton tooling, approximate math policy, implementation ordering, and the definition of “FULL” Triton support.

The target architecture is:

```text
Triton / TTIR
  -> standard MLIR value surface
       vector + memref + arith + math + scf/cf
       plus tiny vc4value launch/policy plumbing
  -> vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> artifacts/runtime
  -> real VC4 hardware
```

The main deliverable of this document is the **staged plan**. Each stage is concrete enough that, after the post-P13 context collection is uploaded and reviewed, a fresh implementation chat could begin generating rigorous phase prompt packages from it.

The implementation philosophy is:

```text
plan the full taxonomy first
  -> implement a narrow value-layer vertical slice
  -> get real TTIR elementwise to hardware quickly
  -> iteratively expand one coherent value capability
  -> prove value -> vc4kernel -> ssavc4 -> scheduled vc4 -> hardware
  -> add the corresponding TTIR bridge/support/reject lock
  -> prove real TTIR -> value -> hardware when executable
  -> converge asymptotically toward full Triton coverage
```

This document is not itself an implementation package. It is a planning and steering artifact.

---

## 1. Source basis

This report is based on:

- `VC4_VECTOR_TRITON_DESIGN_DECISIONS_REPORT.md`
- `VC4_VECTOR_TRITON_PRE_P9_HANDOFF.md`
- the uploaded pre-P9 repo/context zip
- the VideoCore IV 3D Architecture Reference Guide
- the current design discussion in this chat

The earlier design report locked the major architectural split: real TTIR should lower into a standard value layer, and that value layer should lower into `vc4kernel`, not directly into SSAVC4 or scheduled VC4.

The handoff defines the compiler stack, lower-half discipline, verification workflow, and hard boundaries such as:

```text
no direct VC4KernelToVC4
no VC4Tile resurrection
no producer dialect ops inside verified vc4kernel
no Triton direct-to-vc4kernel first
hardware verification mandatory for executable semantics
```

The architecture guide is the hardware source of truth for QPU SIMD-16 execution, TMU memory lookup, VPM/VDR/VDW data movement, SFU behavior, pack/unpack, register hazards, VPM access modes, performance counters, and V3D/QPU launch mechanisms.

---

## 2. Definition of “FULL” Triton support

For this project, **FULL Triton lowering** means:

```text
Every relevant Triton / TTIR construct falls into exactly one of:

A. supported:
   it lowers through the value surface to vc4kernel and hardware;

B. permanently rejected:
   it deterministically rejects, with a compelling hardware/semantic proof
   explaining why VC4 cannot support that specific thing.
```

This is intentionally ambitious. It is an asymptotic target, not a claim that every Triton feature will work immediately.

A crucial distinction:

```text
temporary reject:
  not implemented yet, but hardware/planner support seems plausible.
  These should shrink over time.

permanent reject:
  there is a specific semantic or hardware impossibility.
  These require proof, not vibes.
```

So the long-term target profile should not casually say “unsupported” for difficult features. It should classify each feature as:

```text
supported now
supportable later
supportable only with caveat/approximation
requires emulation path
permanent reject with proof
```

Only the final category satisfies the “cannot support” side of the FULL definition.

---

## 3. Locked architecture

The architecture is now locked:

```text
real Triton source
  -> real emitted TTIR / tt dialect
  -> standard value surface
       func + vc4value + vector + memref + arith + math + scf/cf
  -> verified vc4kernel
  -> ssavc4
  -> scheduled vc4
  -> qasm / C launcher / manifest
  -> libpi runtime
  -> Raspberry Pi VC4 hardware
```

`vc4kernel` is the target-planning dialect. It chooses VC4 execution mechanisms:

```text
fragments
predicates
TMU
VDR
VPM
VDW
barriers
resource metadata
SFU policy
pack/unpack modes
dynamic coordinates
core CFG
```

The value surface is the producer-facing layer. It should express ordinary value semantics:

```text
memory
vectors
masks
reductions
contractions
scalar arithmetic
math
structured control flow
```

It should not expose TMU/VDR/VDW/VPM directly.

Important precision:

```text
No vector dialect ops are legal inside verified vc4kernel.

But vc4kernel may use vector<16xi32> / vector<16xf32> /
similar vector types as target fragment carrier types.
```

---

## 4. Locked design decisions

### 4.1 `vc4value` exists, but stays tiny

`vc4value` fills the gap left by not using `gpu`. It gives the value layer the launch/policy things `vector` does not provide, while avoiding SIMT semantics.

Initial op budget:

```mlir
vc4value.program_id   {axis = 0|1|2} : index
vc4value.num_programs {axis = 0|1|2} : index
```

Initial attrs:

```text
vc4value.kernel
vc4value.grid_rank
vc4value.math_policy / approximate policy marker if needed
vc4value.target_profile if needed
```

Explicitly forbidden from `vc4value`:

```text
memory ops
tile ops
VPM/TMU/VDR/VDW ops
fragment ops
lane ops
barrier ops unless later proven absolutely necessary
```

If `vc4value` starts becoming a second tile dialect, that is a design failure.

### 4.2 Program ID lineage

Natural lineage:

```text
tl.program_id / tt.get_program_id
  -> vc4value.program_id
  -> vc4kernel.program_id
  -> lower-half logical request identity
```

And:

```text
tl.num_programs / tt.num_programs
  -> vc4value.num_programs
  -> vc4kernel.num_programs
  -> launch grid metadata
```

Program IDs are **logical launch-grid coordinates**, not physical QPU IDs.

For a linear request stream:

```text
linear_pid = logical request id

pid0 = linear_pid % grid0
pid1 = (linear_pid / grid0) % grid1
pid2 = linear_pid / (grid0 * grid1)

num_programs(axis) = grid[axis]
```

V1 uses axis 0. GEMM and tiled kernels use axes 0/1/2 later without redesign.

### 4.3 Kernel wrapper

Use `func.func` plus attrs at the value layer:

```mlir
func.func @kernel(%x: memref<?xf32, #vc4value.global>,
                  %y: memref<?xf32, #vc4value.global>,
                  %n: index)
    attributes {vc4value.kernel, vc4value.grid_rank = 1} {
  %pid = vc4value.program_id {axis = 0} : index
  ...
  return
}
```

Lowering produces:

```mlir
vc4kernel.kernel @kernel(...)
```

The verified `vc4kernel` body contains only legal target-planning IR and core CFG.

### 4.4 Memory representation

The value layer uses logical `memref`.

Recommended spelling:

```mlir
memref<?xf32, #vc4value.global>
memref<?xi32, #vc4value.global>
memref<?x?xf32, strided<[?, 1], offset: 0>, #vc4value.global>
```

The exact syntax can be finalized during implementation, but the semantic decision is locked:

```text
value layer:
  memref rank, element type, shape, layout, stride, memory space, access attrs

vc4kernel:
  raw i32 base pointer uniforms + scalar sizes/strides
```

This keeps enough structure for the planner to choose TMU versus VDR/VPM versus VDW.

### 4.5 Value surface does not expose VC4 data paths

The value surface expresses:

```text
memref
vector.transfer_read
vector.transfer_write
vector.gather later
vector.contract later
vector.reduction
vector.shuffle / transpose later
scf/cf
```

The planner chooses:

```text
TMU direct memory lookup
VDR -> VPM DMA load
QPU VPM read/write
VDW DMA store
register-staged VDW store
```

This is required because VC4 load/store facilities are not symmetric. Loads may naturally use TMU or VDR; stores generally need VDW and inactive-preserve policy. VPM, VDR, and VDW also have distinct setup fields and mode limitations.

### 4.6 Lane/arange representation

Triton:

```python
offs = pid * BLOCK + tl.arange(0, BLOCK)
```

Value layer:

```mlir
%lane = vector.step : vector<16xindex>
%base_vec = vector.splat %base : vector<16xindex>
%idx = arith.addi %base_vec, %lane : vector<16xindex>
```

Use `vector.step` as the canonical lane vector. Do not invent a `vc4value.lane_id` op unless implementation proves `vector.step` is inadequate.

For V1:

```text
BLOCK_SIZE = 16
one Triton block fragment maps naturally to one QPU SIMD-16 fragment
```

For larger blocks:

```text
split into multiple vector<16> fragments or loops of vector<16>
```

### 4.7 Transfers and masks

Use standard vector transfer ops:

```text
vector.transfer_read
vector.transfer_write
```

The value-to-`vc4kernel` planner classifies masks:

```text
full
empty
tail
rect
sparse
unknown
```

Baseline lowering:

```text
full/tail transfer_read:
  TMU safe-offset inactive zero
  or VDR/VPM if structured tile reuse is recognized

full/tail/rect transfer_write:
  VDW inactive-preserve path when representable

sparse compute mask:
  predicate/select/reduction path if safe

sparse store:
  deterministic reject unless a later hardware-proven sparse-store path exists
```

### 4.8 Gather/scatter distinction

Loads and stores are not symmetric.

Triton pointer-tensor loads often become gather-like patterns. VC4 TMU direct-address memory lookup makes many gather **loads** plausible.

Sparse stores remain fundamentally harder.

Staged policy:

```text
contiguous/tail load:
  vector.transfer_read -> TMU or VDR/VPM

gather load:
  vector.gather -> TMU direct-address load, later phase

contiguous/tail/rect store:
  vector.transfer_write -> VDW preserve

arbitrary scatter/sparse store:
  reject unless transformed into dense/rect store
  or unless future hardware proof adds sparse-store support
```

### 4.9 Math policy

Default project direction:

```text
lower more, reject less, then add guardrails
```

But not by cheating.

The IR must distinguish:

```text
strict/exact
finite-tree
triton_fast
approx_sfu
```

Policy:

```text
handwritten value fixtures:
  strict unless explicitly marked

real TTIR import:
  preserve Triton's fast/approx intent where present
  lower approximate forms with explicit metadata/caveat
  do not silently use approximate SFU for precise source semantics
```

Long-term, this supports “more of Triton” while keeping diagnostics and manifests honest.

### 4.10 Control flow

`scf` may exist in value-layer input.

Verified `vc4kernel` must not contain raw `scf`.

Pipeline:

```text
value input:
  scf.for / scf.if allowed in surface profile

canonicalization:
  scf -> cf
  loop-carried values -> block args

vc4kernel:
  cf.br / cf.cond_br-style CFG and block args
```

Old VC4Tile-era `scf -> cf` work may be mined for fixtures or code patterns, but VC4Tile itself must not be revived.

### 4.11 Contract/GEMM

`tt.dot` should lower through:

```text
tt.dot
  -> vector.contract
  -> VC4 tile/data planner
  -> vc4kernel VDR/VPM/QPU fragment/VDW plan
```

Do not invent `vc4value.dot` or `vc4tile.contract`.

First natural VC4 contract shape:

```text
one output row
16 output columns across SIMD lanes
K staged in small chunks
```

Conceptually:

```text
C[pid_m, pid_n * 16 + lane]
  += A[pid_m, k] * B[k, pid_n * 16 + lane]
```

The initial GEMM path should be staged:

```text
1x16x4
1x16x8 / 1x16x16
2x16xK
4x16xK
tail M/N
larger K loops
cooperative multi-QPU tiles later
```

### 4.12 Triton tooling

Recommended architecture:

```text
vc4-triton-import
  input: real emitted TTIR MLIR
  output: vc4 value-surface MLIR
```

The core VC4 compiler should not require Triton to run ordinary `check-vc4`.

For reproducibility:

```text
check in:
  examples/foo.py
  generated/foo.ttir.mlir
  expected/foo.vc4value.mlir

optional developer script:
  regenerate TTIR from pinned Triton version
```

The demo target should look like something ML systems / frontier-lab kernel people care about:

```text
real Triton source
  -> real TTIR
  -> Pi GPU hardware result
```

A future CUDA Tile flag that emits TTIR is a secondary bonus. If it emits compatible TTIR, it can use the same importer path; it should not influence the architecture now.

---

## 5. Verification and anti-cheat policy

Every executable semantic stage needs hardware proof.

Prompt-package shape after post-P13:

```text
1. inventory/design lock
2. narrow implementation
3. targeted hardware proof
4. migration/audit lock
5. mixed acceptance final gate
```

No stage should:

```text
weaken verifier diagnostics
weaken CPU oracle
remove sentinels
reduce adversarial dimensions
special-case fixture names/paths/public_name/status
substitute generated output
bypass vc4kernel -> ssavc4 -> vc4
revive VC4Tile
reshape natural kernels to hide lower-half bugs
```

Final acceptance should continue the P8.5 philosophy:

```text
isolated fixtures remain for triage
mixed fixtures are routine final acceptance
new features add mixed interaction coverage
```

---

## 6. Performance stance

Correctness comes first.

But the design must remain performance-conscious:

```text
do not choose a value surface that prevents VPM tiling later
do not flatten everything into scalarized elementwise code
do not hide memory structure too early
do not make program_id/grid metadata impossible to use for 2D tiles
do not make approximate math policy impossible to exploit
```

Once elementwise and GEMM paths exist, begin collecting:

```text
rough runtime
QPU instruction counts
TMU stall counters
VPM/VCD/VDW stall counters
L2 hit/miss counters
```

Performance counters should initially be observability, not pass/fail gates.

---

# 7. Staged Plan

The rest of this document is the planned sequence from post-P13 context refresh to full Triton lowering.

Each phase is self-contained and should be converted into prompt packages only after the post-P13 context collection is uploaded and reviewed.

---

## Phase 0 — Post-P13 rebaseline

### Goal

Replace all pre-P13 assumptions with the final accepted/rejected `vc4kernel` surface.

### Inputs

```text
post-P13 context zip
final vc4kernel strict specification
final support matrix
final mixed acceptance policy
P9 exact subword mode table
P10 SFU/math contract
P11 rotate/shuffle table
P12 dynamic VPM/VDR/VDW coordinate table
P13 final surface lock/audits
```

### Work

Produce an updated understanding report:

```text
what changed since pre-P9/P8.5
final accepted vc4kernel features
final rejected vc4kernel features
final hardware-proven subword/SFU/shuffle/VPM modes
updated lower-half hazards
updated mixed acceptance suite
```

### Deliverables

```text
.vc4_auto/vector_triton_post_p13_rebaseline/REPORT.md
updated staged plan notes if needed
READY_FOR_VALUE_SURFACE_PACKAGES=YES/NO
```

### Gate

No implementation prompt packages until this phase says the context is coherent.

---

## Phase 1 — Full value/Triton taxonomy and target-profile spec

### Goal

Define the full classification space before implementing V1.

This prevents an elementwise-only V1 from baking in wrong assumptions.

### Scope

Classify every major family:

```text
launch/program_id
num_programs/grid metadata
kernel wrapper
memref ABI
memory spaces
vector types/ranks
masks
transfer_read/write
gather/scatter
arith
math
subword
pack/unpack
reductions
contract/dot
scf/cf
shuffle/transpose
block pointers
tensor descriptors
cooperative blocks
num_warps/num_stages
approx math
diagnostics
permanent reject proof format
```

### Important policy

Define “full” as:

```text
supported
supportable but staged
supportable with caveat/approximation
supportable by slow emulation
permanent reject with proof
```

Do not label a feature permanently rejected merely because it is hard.

### Deliverables

```text
compiler/docs/vc4_value_surface_spec.md
compiler/docs/vc4_ttir_target_profile.md
compiler/docs/vc4_ttir_reject_proof_policy.md
compiler/docs/vc4_value_to_vc4kernel_planning.md
```

### Gate

Docs only. No lowering yet.

---

## Phase 2 — Tiny `vc4value` launch/policy layer

### Goal

Create the minimal value-layer gap-filler.

### Accepted initial ops

```mlir
vc4value.program_id   {axis = 0|1|2} : index
vc4value.num_programs {axis = 0|1|2} : index
```

### Accepted initial attrs

```text
vc4value.kernel
vc4value.grid_rank
vc4value.math_policy if needed
vc4value.target_profile if needed
```

### Explicitly forbidden

```text
vc4value.load
vc4value.store
vc4value.tile
vc4value.vpm
vc4value.tmu
vc4value.vdr
vc4value.vdw
vc4value.fragment
vc4value.lane_id unless vector.step fails
```

### Deliverables

```text
dialect td files
parser/printer/verifier
docs
lit tests
audit proving op set is tiny
```

### Gate

`vc4value` must remain launch/policy plumbing, not a second producer dialect.

---

## Phase 3 — Value-surface verifier and audit skeleton

### Goal

Make the value surface a real contract.

### Allowed dialects

```text
builtin
func
vc4value
vector
memref
arith
math
scf
cf
```

### Forbidden dialects

```text
tt
ttg
gpu
linalg initially
nvgpu
nvvm
rocdl
spirv
iree
stablehlo/mhlo
vc4kernel mixed into value input
ssavc4
vc4
```

### Work

Add:

```text
--vc4-verify-value-surface
value surface support matrix
forbidden dialect audit
feature status audit
diagnostic tests
```

### Deliverables

```text
compiler/docs/vc4_value_surface_support_matrix.json
compiler/test/ValueSurface/...
```

### Gate

The value layer must have the same contract discipline as `vc4kernel`.

---

## Phase 4 — Kernel wrapper, launch ABI, and memref ABI

### Goal

Define how a value-surface kernel becomes a real hardware-launchable kernel.

### Lock

```text
func.func wrapper attrs
grid_rank metadata
program_id axis decomposition
num_programs binding
memref global memory-space spelling
base pointer uniform order
dynamic size/stride argument order
readonly/writeonly/inout attrs
index -> i32 lowering policy
```

### Initial V1 subset

```text
grid_rank = 1
memref<?xf32, #vc4value.global>
memref<?xi32, #vc4value.global>
contiguous layout
index sizes lowered to i32
```

### Later-compatible design

Must already allow:

```text
rank-2 memrefs
strides
2D/3D program_id
GEMM tile coordinates
dynamic sizes
```

### Deliverables

```text
ABI docs
lit tests for wrapper legality
lit tests for uniform order / manifest metadata
negative tests for illegal memref forms
```

### Gate

No value-to-`vc4kernel` memory lowering until ABI is explicit.

---

## Phase 5 — V1 handwritten value elementwise to hardware

### Goal

First real value-surface vertical slice:

```text
func/vc4value/vector/memref/arith
  -> vc4kernel
  -> ssavc4
  -> vc4
  -> hardware
```

### Supported value features

```text
vc4value.program_id(0)
vc4value.num_programs(0)
vector.step : vector<16xindex>
vector.splat
vector.create_mask / canonical tail mask
vector.transfer_read contiguous/tail
vector.transfer_write contiguous/tail
vector<16xi32>
vector<16xf32>
arith.addi/subi/muli where legal
arith.addf/subf/mulf where legal
cmp/select
constants/splats
```

### Planner behavior

```text
transfer_read:
  TMU safe-offset inactive zero

transfer_write:
  VDW inactive preserve

arith/cmp/select:
  vc4kernel fragment ops

mask:
  vc4kernel pred<16>
```

### Hardware fixtures

```text
copy f32
saxpy f32
masked tail f32
i32 add/select
f32 cmp/select
sentinel-preserve store
```

### Deliverables

```text
value-to-vc4kernel lowering pass
handwritten value fixtures
hardware candidate generation
CPU oracle + sentinel checks
lit tests
support matrix update
```

### Gate

This phase proves handwritten value IR reaches hardware.

---

## Phase 6 — Real TTIR inventory and importer skeleton

### Goal

Bring in real Triton, but only as far as inventory and elementwise import scaffolding.

### Work

```text
pin Triton version/commit
generate real TTIR from tiny Triton kernels
check in source.py + emitted .ttir.mlir
inspect actual tt dialect forms
design importer architecture
avoid fake TTIR
avoid TTGIR/NVIDIA backend IR
```

### Initial TTIR concepts to inventory

```text
tt.func
tt.return
tt.get_program_id
tt.arange
tt.load with mask/other
tt.store with mask
broadcast/splat
arith
cmp
select
constexpr block sizes
```

### Tool recommendation

```text
vc4-triton-import input.ttir.mlir -o output.vc4value.mlir
```

Normal VC4 CI should not require Triton regeneration.

### Deliverables

```text
compiler/docs/vc4_real_ttir_inventory.md
tools/vc4-triton-import skeleton or design
examples/triton/elementwise/*.py
examples/triton/elementwise/*.ttir.mlir
lit test: TTIR -> value IR textual conversion
```

### Gate

Importer must use real TTIR dialect registration/parsing, not regex.

---

## Phase 7 — Real TTIR elementwise smoke to hardware

### Goal

Prove the full big-picture pipeline quickly.

### Pipeline

```text
real Triton source
  -> real TTIR
  -> vc4-triton-import
  -> vc4 value IR
  -> Phase 5 value-to-vc4kernel path
  -> ssavc4
  -> vc4
  -> hardware
```

### Keep tiny

```text
one vector add/copy kernel
one saxpy or relu/select kernel
tail mask included
```

### Deliverables

```text
end-to-end script
checked-in generated TTIR
checked-in value IR
hardware fixtures
README demo instructions
```

### Gate

At least one real Triton source kernel lowers to Pi GPU hardware and passes CPU oracle.

This is the first “cool demo” milestone.

---

## Phase 8 — Control flow and loop boundary

### Goal

Allow value input to use structured control flow while preserving verified `vc4kernel` restrictions.

### Supported

```text
scf.for
scf.if
loop-carried values
canonical scf -> cf
block args
natural loops
```

### Use cases

```text
grid-stride elementwise loops
K loops for reductions
K loops for GEMM
small looped kernels
```

### Work

```text
decide whether value-to-vc4kernel requires prior convert-scf-to-cf
mine old VC4Tile-era scf/cf code/tests if useful
add verifier rules
reject unsupported irreducible/cyclic forms
```

### Hardware fixtures

```text
looped saxpy
grid-stride copy
small loop-carried accumulator
```

### Gate

No raw `scf` survives into verified `vc4kernel`.

Phase 8 final readiness points to Phase 8.5 TTIR control-flow
bridge/support/reject lock, not Phase 9.

---

## Phase 8.5 — TTIR control-flow bridge/support/reject lock

### Goal

Inventory real Triton-emitted TTIR control-flow forms against the Phase 8
value `scf`/`cf` boundary, then either bridge matching forms to value IR or
lock deterministic staged/reject classifications with exact reasons.

### Initial classification targets

```text
Triton compile-time/meta control flow:
  frontend-specialization; no runtime value control-flow import required

tl.range / emitted TTIR loop forms:
  Phase 8.5 inventory target

scalar cf/scf branch forms:
  lowerable if they match the Phase 8 value-cf subset, otherwise staged

vector/per-lane branch conditions:
  deterministic reject as control flow; must become masks/selects

ttg/triton_gpu/nvgpu/nvvm backend control flow:
  reject at TTIR frontend boundary

warp-specialized or async-partition control metadata:
  inspect in Phase 8.5; ignore only if proven semantic-no-op for emitted TTIR
```

### Gate

No broad TTIR control-flow support claim. If executable forms are accepted, they
must lower through `TTIR -> value -> vc4kernel -> ssavc4 -> scheduled vc4` and
hardware when required. `READY_FOR_TRITON` remains `NO`.

---

## Phase 9 — Mask classifier and richer memory legality

### Goal

Centralize memory legality.

### Classifier

```text
full
empty
tail
rect
sparse
unknown
```

### Supported after this phase

```text
full/tail transfer_read
full/tail transfer_write
empty no-op stores
rect transfer patterns where final vc4kernel supports them
general compute masks for select/cmp
```

### Rejected

```text
unknown store mask
sparse store mask
unsupported rect store
```

### Deliverables

```text
mask classifier library
diagnostic tests
value support matrix update
hardware fixtures for tail/rect/empty
negative sparse-store tests
```

### Gate

No memory lowering should pattern-match masks ad hoc after this phase.

Phase 9 final readiness points to Phase 9.5 TTIR mask/memory-legality
bridge/support/reject lock, not Phase 10.

---

## Phase 9.5 — TTIR mask/memory-legality bridge/support/reject lock

### Goal

Map real TTIR mask and memory forms onto the Phase 9 value mask classifier and
memory-legality categories, then either bridge executable forms or lock staged
and deterministic reject classifications with exact reasons.

### Targets

```text
canonical tail masks:
  bridge to value tail masks where they match the classifier

full/empty masks:
  bridge to value full/empty forms where structurally proven

rectangular memory masks:
  staged or bridge only where Phase 9 value legality proves dense/rect effects

sparse/unknown store masks:
  deterministic reject unless a future hardware-proven emulation path exists

non-affine pointer tensors:
  staged for Phase 10/10.5 gather or strided memory classification
```

### Gate

TTIR memory import must use structural mask and pointer analysis, not source
names or printed-IR substrings. `READY_FOR_TRITON` remains `NO`.

---

## Phase 10 — Memory expansion: gather loads and strided transfers

### Goal

Support more Triton-like pointer patterns without allowing illegal stores.

### Add

```text
affine strided transfer_read
affine strided transfer_write when dense/rect
vector.gather for 32-bit loads
TMU direct-address gather load
```

### Still reject

```text
vector.scatter
arbitrary sparse stores
non-affine stores that cannot prove dense/rect/tail
```

### TTIR connection

Start accepting tensor-of-pointers loads when they reconstruct to:

```text
base + affine vector offset
base + gather vector offset
```

### Hardware fixtures

```text
strided load + dense store
gather load + dense store
negative scatter store
```

### Gate

This expands load coverage without violating VDW sparse-store rules.

---

## Phase 10.5 — TTIR gather/strided memory bridge

### Goal

Connect Phase 10 value gather-load and strided-transfer legality to real TTIR
pointer-tensor forms, while preserving the sparse-store reject boundary.

### Gate

Executable TTIR gather or strided memory forms must lower through the standard
value path and hardware when claimed. Unsupported scatter or sparse-store forms
must reject deterministically with the first unsupported boundary named.

---

## Phase 11 — Reductions

### Goal

Support reduction-style kernels through `vector.reduction`.

### Value support

```text
i32 add/min/max/and/or/xor
f32 add/min/max under explicit finite-tree or approx policy
tail-masked reductions
```

### Lowering

```text
vector.reduction
  -> vc4kernel.fragment_reduce
```

### TTIR connection

```text
tt.reduce
  -> vector.reduction
```

### Hardware fixtures

```text
i32 sum
i32 max
f32 finite sum
f32 finite max
tail-masked row reduction
```

### Gate

F32 reductions must carry explicit finite/approx policy. No silent strict-IEEE reassociation.

---

## Phase 12 — TTIR reductions smoke

### Goal

Connect the Phase 11 value reduction path to real Triton before moving to the
next major value capability.

### Triton kernels

```text
row sum
row max
small reduction with mask
```

### Pipeline

```text
real Triton source
  -> TTIR
  -> value IR with vector.reduction
  -> vc4kernel
  -> hardware
```

### Gate

Real TTIR `tt.reduce` reaches hardware for at least one row-style reduction.

---

## Phase 13 — Math and SFU policy

### Goal

Support useful Triton math while being honest about approximation.

### Value support

```text
math.exp
math.log
math.rsqrt
math.sqrt
reciprocal/div patterns
```

### Policies

```text
strict/exact:
  exact sequence or reject

finite:
  finite-only lowering accepted

triton_fast:
  preserve Triton fast intent

approx_sfu:
  use VC4 SFU with caveat metadata
```

### Lowering

```text
math.exp/log/rsqrt/recip under approx policy
  -> vc4kernel SFU ops

precise math
  -> exact sequence if implemented
  -> deterministic diagnostic otherwise
```

### Hardware fixtures

```text
approx recip
approx rsqrt
approx exp
approx log
strict-mode negative tests
manifest caveat checks
```

### Gate

Approximate math must be explicit in IR, diagnostics, or manifest.

---

## Phase 14 — TTIR math smoke

### Goal

Connect real Triton fast math to the Phase 13 value math policy before moving
to the next major value capability.

### Supported first

```text
fast exp/log/rsqrt/recip-like patterns
activation-style kernels
approx caveat metadata
```

### Reject or defer

```text
precise math requiring exact semantics not yet implemented
```

### Hardware fixtures

```text
activation-style elementwise
approx math + tail store
```

### Gate

This is where “lower more, reject less, guardrail later” starts paying off.

---

## Phase 15 — Subword types and pack/unpack

### Goal

Use the final P9 mode table for i8/i16/f16-style value support.

### Value features

```text
memref<?xi8>
memref<?xi16>
memref<?xf16> where legal
vector<16xi8/i16/f16> where legal
extend/trunc
bitcast
pack/unpack
```

### Critical split

Handle separately:

```text
fragment pack/unpack
VPM QPU read modes
VPM QPU write modes
VDR DMA modes
VDW DMA modes
TMU returned packed data / r4 unpack where relevant
```

Do not assume symmetry.

### Hardware fixtures

```text
subword load roundtrip
subword store preserve
subword VPM read/write
subword VDR load
subword VDW store
negative unsupported modes from exact table
```

### Gate

Every accepted subword path must cite the exact accepted mode table entry.

---

## Phase 16 — TTIR subword smoke

### Goal

Connect real Triton subword kernels to the Phase 15 proven value paths before
moving to the next major value capability.

### Supported first

```text
i8/i16 loads
i8/i16 stores where legal
casts/extends where exact or policy-accepted
f16/bf16 only if final surface proves support
```

### Hardware fixtures

```text
subword copy
subword widen-compute-store
subword tail mask
```

### Gate

No generic “Triton int8 support” claim. Each mode is table-backed.

---

## Phase 17 — Shuffle, rotate, transpose

### Goal

Support lane movement needed by reductions, layout transforms, and GEMM.

### Value features

```text
vector.shuffle
vector.transpose
vector.extract/insert patterns
static rotate
dynamic rotate where accepted
```

### Lowering

```text
simple rotate/shuffle
  -> vc4kernel rotate/shuffle

unsupported arbitrary permutation
  -> deterministic reject
  -> or scalarized/emulated only if proven
```

### Hardware fixtures

```text
static rotate
dynamic rotate
shuffle + reduction
transpose-like VPM preparation
```

### Gate

Unsupported permutations must produce precise diagnostics.

---

## Phase 18 — VPM tile transfer planner

### Goal

Begin exploiting the full VC4 data-movement work for tiled ML kernels.

### Input remains standard value IR

```text
rank-2 memrefs
strided layouts
scf loops
vector.transfer_read/write
subview-like patterns
rect masks
affine-ish index expressions
```

### Planner emits

```text
VDR -> VPM
QPU VPM read/write
VDW
barriers/waits/resources
dynamic coordinates/pitch
VPM row allocation
```

### First targets

```text
2D tile copy
blocked GEMV
VPM transpose
rect store preserve
```

### Hardware fixtures

```text
blocked GEMV value-layer fixture
VPM tile load/store fixture
dynamic pitch/coordinate fixture
transpose-through-VPM fixture
```

### Gate

This phase proves the value surface can express tile-like structure without becoming a tile DSL.

---

## Phase 19 — `vector.contract` row-fragment GEMM

### Goal

First contraction lowering.

### Input

```text
vector.contract
```

### First VC4-natural shape

```text
BLOCK_M = 1
BLOCK_N = 16
BLOCK_K = small, initially 4
```

### Conceptual lowering

```text
A scalar / short K load
B vector fragment across 16 columns
FMA-style accumulation with vector<16xf32>
VDW dense/tail store for C row fragment
```

### Hardware fixtures

```text
1x16x4 GEMM
tail-N GEMM
known CPU oracle
sentinel-preserve C store
```

### Gate

This is the first real destination for `tt.dot`.

---

## Phase 20 — TTIR dot smoke

### Goal

Connect real Triton `tt.dot` to the proven `vector.contract` path.

### Accepted first subset

```text
static BLOCK_M/N/K from accepted set
affine pointer patterns
supported element types
explicit precision policy
no tensor-core assumptions
no implicit TF32
```

### Pipeline

```text
tt.dot
  -> vector.contract
  -> Phase 19 contract lowering
  -> hardware
```

### Hardware fixture

```text
real Triton matmul microkernel, tiny shape
```

### Gate

A real Triton dot/matmul-style kernel reaches Pi GPU hardware.

---

## Phase 21 — GEMM shape expansion

### Goal

Expand GEMM one natural stage at a time.

### Recommended order

```text
1x16x4
1x16x8
1x16x16
2x16xK
4x16xK
tail M
tail N
larger K loops
```

### Work per stage

```text
one value fixture
one TTIR fixture if applicable
one hardware oracle
one mixed interaction fixture
support matrix update
```

### Gate

Do not jump to cooperative multi-QPU GEMM until single-QPU row-fragment GEMM is boring.

---

## Phase 22 — Cooperative block planning

### Goal

Use multiple QPUs for one logical Triton program tile when useful.

### Important boundary

The value IR still does not expose SIMT warps.

Triton metadata:

```text
num_warps
```

is planner metadata, not a source-level thread model.

### Planner creates internally

```text
warps_per_block
logical worker id
barriers
shared VPM layout
semaphore resources
resource metadata
```

### Hardware fixtures

```text
cooperative GEMM tile
barrier/VPM reuse
multi-QPU correctness
```

### Gate

Cooperative planning must not leak physical QPU IDs into the value surface.

---

## Phase 23 — VPM pipelining and double buffering

### Goal

Improve tiled-kernel planning without changing value semantics.

### Add

```text
double-buffered VPM tiles
overlap VDR load with compute where safe
VDW writeback staging
resource accounting
```

### Performance observability

Start collecting:

```text
QPU cycles
TMU stalls
VPM/VCD/VDW stalls
L2 hits/misses
instruction counts
```

### Gate

Correctness remains mandatory. Performance counters guide design but do not replace CPU oracle/sentinel checks.

---

## Phase 24 — Block pointers and tensor descriptors

### Goal

Support richer Triton memory structure after affine/tensor-pointer paths work.

### Add

```text
Triton block pointers
boundary checks
padding options
rectangular tiles
tensor-descriptor-like forms if they lower cleanly
```

### Lower to value IR as

```text
structured memref/vector.transfer patterns
```

not direct VPM ops.

### Gate

Block pointers are added after the VPM tile planner and contract path are already proven, so they preserve structure rather than becoming a second tiling mechanism.

---

## Phase 25 — Broader Triton memory profile

### Goal

Expand real TTIR memory coverage.

### Support

```text
contiguous tensor-of-pointers
affine strided pointers
gather loads
block-pointer loads
dense/tail/rect stores
boundary-check/padding forms that map to masks/pad values
```

### Reject with proof or defer

```text
arbitrary sparse stores
atomics until a real emulation/hardware plan exists
unsupported memory ordering semantics
cache modifiers that source requires semantically
```

### Note

Cache/eviction hints can often be ignored with a note if they are non-semantic hints. They should reject only if the source requires semantics VC4 cannot provide.

---

## Phase 26 — Frontier-lab-relevant Triton kernel corpus

### Goal

Move from microfeatures to recognizable ML kernels.

### Initial corpus

```text
elementwise add/copy/saxpy
relu/select/activation
row sum
row max
softmax-like row kernel
GEMV
tiny matmul
larger staged matmul
subword copy/widen-compute
approx math activation
```

### Later corpus

```text
RMSNorm-like row kernel
LayerNorm-like row kernel
attention score microkernel
softmax + matmul fragments
quantized matmul pieces where subword support allows
```

### Deliverables

```text
examples/triton/*.py
checked-in TTIR
expected value IR
hardware runner configs
README demo path
```

### Gate

These should look like kernels ML systems people recognize, not artificial compiler-only kernels.

---

## Phase 27 — Source-to-hardware driver

### Goal

Make the demo path clean.

### Developer workflow

```bash
python examples/triton/saxpy.py --emit-ttir
vc4-triton-import saxpy.ttir.mlir -o saxpy.vc4value.mlir
vc4-opt saxpy.vc4value.mlir \
  --convert-vc4-value-to-vc4kernel \
  --convert-vc4kernel-to-ssavc4 \
  ...
vc4-codegen ...
run hardware candidate
```

Possible convenience wrapper later:

```bash
vc4-triton-compile examples/triton/saxpy.py --run-hardware
```

### Requirements

```text
real Triton source
real emitted TTIR
auditable intermediate value IR
hardware PASS
```

### Gate

Do not hide the intermediate IRs. The demo is stronger when people can see each layer.

---

## Phase 28 — Full TTIR operation/profile closure

### Goal

No unknowns in the chosen Triton/TTIR target profile.

### Work

For every encountered TTIR op/form:

```text
classify
lower
lower with caveat
emulate
or permanently reject with proof
```

### Deliverables

```text
compiler/docs/vc4_ttir_supported_rejected_matrix.json
compiler/docs/vc4_ttir_permanent_reject_proofs.md
diagnostic tests
importer tests
value lowering tests
hardware mixed suite
```

### Proof standard for permanent reject

Each permanent reject must identify:

```text
source construct
required semantics
candidate hardware mechanisms considered
why each fails
whether slow emulation is impossible or merely not implemented
final diagnostic text
```

### Gate

No “unsupported because not implemented” in the permanent-reject table.

---

## Phase 29 — Value/Triton mixed acceptance lock

### Goal

Make upper-stack final acceptance resemble the hardened `vc4kernel` process.

### Mixed fixtures

```text
TTIR elementwise + tail + TMU + VDW
TTIR gather load + dense store
TTIR loop + reduction + spill
TTIR approximate math + caveat manifest
TTIR subword + VPM/VDW
TTIR shuffle + reduction
TTIR VPM tile GEMV
TTIR dot/GEMM
cooperative GEMM
negative sparse store
negative exact math violation
negative unsupported subword mode
negative permanent-reject diagnostic
```

### Gate

Routine final acceptance uses mixed fixtures plus audits/lit/checks, not every isolated fixture.

---

## Phase 30 — Optimization and broadening

### Goal

After correctness and full-profile classification, improve usefulness.

### Expand

```text
more block sizes
more GEMM shapes
better VPM tiling
double buffering
multi-QPU cooperative plans
more subword modes
more math patterns
better TTIR canonicalization
more Triton tutorial kernels
performance-counter guided planning
```

### Rule

Every new executable semantic needs hardware proof.

---

# 8. Compact phase sequence

```text
0  Post-P13 rebaseline
1  Full value/Triton taxonomy and target-profile spec
2  Tiny vc4value launch/policy layer
3  Value-surface verifier and audit skeleton
4  Kernel wrapper, launch ABI, and memref ABI
5  V1 handwritten value elementwise to hardware
6  Real TTIR inventory and importer skeleton
7  Real TTIR elementwise smoke to hardware
8  Control flow and loop boundary
8.5 TTIR control-flow bridge/support/reject lock
9  Mask classifier and richer memory legality
9.5 TTIR mask/memory-legality bridge/support/reject lock
10 Memory expansion: gather loads and strided transfers
10.5 TTIR gather/strided memory bridge
11 Reductions
12 TTIR reductions smoke
13 Math and SFU policy
14 TTIR math smoke
15 Subword types and pack/unpack
16 TTIR subword smoke
17 Shuffle, rotate, transpose
18 VPM tile transfer planner
19 vector.contract row-fragment GEMM
20 TTIR dot smoke
21 GEMM shape expansion
22 Cooperative block planning
23 VPM pipelining and double buffering
24 Block pointers and tensor descriptors
25 Broader Triton memory profile
26 Frontier-lab-relevant Triton kernel corpus
27 Source-to-hardware driver
28 Full TTIR operation/profile closure
29 Value/Triton mixed acceptance lock
30 Optimization and broadening
```

---

# 9. What remains open

Nothing blocks the staged plan. The remaining open items are implementation-time choices, not architectural uncertainties.

## 9.1 Exact syntax choices

Still to decide during packages:

```text
exact memory-space attr spelling
exact math policy attr spelling
exact support-matrix schema
whether vc4-triton-import is separate binary or vc4-opt mode
```

Recommendation remains:

```text
use #vc4value.global or similarly named memory space
use separate optional vc4-triton-import tool
keep core check-vc4 independent of Triton regeneration
```

## 9.2 Acceptance corpus expansion

The first corpus should be small and real. The later corpus should move toward recognizable ML kernels:

```text
elementwise
reductions
softmax-like row kernels
GEMV
GEMM
RMSNorm/LayerNorm-like kernels
attention fragments
quantized/subword kernels
```

## 9.3 Permanent reject proof culture

This is the biggest philosophical requirement from the new FULL definition. Temporary staging rejects are fine, but final permanent rejects need actual hardware/semantic arguments.

---

# 10. Steering prompt for a fresh implementation chat

```text
You are working on the VC4 compiler project after the final P13 vc4kernel surface lock.

The target architecture is:
  real Triton / TTIR
    -> standard MLIR value surface
         func + tiny vc4value + vector + memref + arith + math + scf/cf
    -> vc4kernel
    -> ssavc4
    -> scheduled vc4
    -> artifacts/runtime
    -> real VC4 hardware

Hard rules:
  - no direct VC4KernelToVC4
  - no VC4Tile resurrection
  - no producer dialect ops inside verified vc4kernel
  - no Triton direct-to-vc4kernel first
  - no TMU/VDR/VDW/VPM ops in the value surface
  - vc4value must stay tiny: program_id, num_programs, and metadata only
  - value memory is memref/vector semantics; the planner chooses TMU/VDR/VPM/VDW
  - vector dialect ops are not legal inside verified vc4kernel, though vc4kernel may use vector<16xT> fragment carrier types
  - sparse stores reject unless hardware-proven
  - approximate math lowers with explicit policy/caveat, not silently
  - executable semantics require hardware proof
  - do not reshape kernels to hide compiler/lower-half bugs

Definition of FULL:
  Every Triton/TTIR construct must eventually be either:
    A. supported/lowered, or
    B. deterministically rejected with a compelling proof that VC4 hardware cannot support the required semantics.

Implementation order:
  Start with the staged plan:
    Phase 0 post-P13 rebaseline
    Phase 1 full taxonomy/spec
    Phase 2 tiny vc4value
    Phase 3 value verifier/audits
    Phase 4 ABI
    Phase 5 handwritten vector elementwise to hardware
    Phase 6 real TTIR inventory/importer
    Phase 7 real TTIR elementwise smoke to hardware
    Phase 8 value control flow
    Phase 8.5 TTIR control-flow bridge/support/reject lock
    Phase 9 value mask/memory legality
    Phase 9.5 TTIR mask/memory-legality bridge/support/reject lock
    Phase 10 value gather/strided memory
    Phase 10.5 TTIR gather/strided memory bridge
    then continue in value/TTIR pairs: reductions, math, subword, shuffle, VPM tiling, vector.contract/GEMM, TTIR dot, block pointers, full profile closure.

For each implementation phase, generate prompt packages in the established project style:
  1. read-only inventory/design lock
  2. narrow implementation
  3. targeted hardware proof
  4. migration/audit lock
  5. mixed final acceptance
```

---

# 11. Summary

The plan is now coherent:

```text
Design the full Triton/value/vc4kernel taxonomy first.
Implement only a narrow elementwise value slice first.
Immediately prove real TTIR source-to-hardware smoke.
Then iteratively expand one coherent value capability, prove it through
hardware when executable, and connect the corresponding real TTIR forms before
moving to the next major value capability.
Keep all hardware-specific planning below the value layer.
Use vc4value only for tiny launch/policy plumbing.
Make “FULL Triton” mean supported or permanently rejected with proof.
```

This gives the project both the near-term demo path and the long-term architecture needed for real ML-kernel credibility on the Pi GPU.
