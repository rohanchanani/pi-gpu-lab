# VC4Tile M5 Full Design: Ergonomic VC4Tile Surface, Copy Planner, Tile Compute Primitives, and 32-Bit-First Hardware-Proven Lowering

**Date:** 2026-05-27  
**Status:** full M5 design draft for milestone-package generation and implementation planning  
**Scope:** M5 ergonomic VC4Tile milestone after M4 and the post-M4/pre-M5 SCF/core-CFG staging work  
**Immediate non-scope:** Triton lowering, IREE/JAX/PyTorch lowering, real sub-32 precision implementation, direct VC4Tile-to-scheduled-VC4 lowering, and any producer-to-SSAVC4 shortcut

---

## 1. Companion documents and how this document should be used

This full M5 design assumes the existence of three focused design documents:

1. `vc4tile_m5_compute_primitives_design.md` / `vc4tile_m5_compute_primitives_design(1).md`  
   Locks the decision that `tile_contract`, `tile_dot`, `tile_matmul`, `tile_reduce`, and related computation primitives belong in the ergonomic VC4Tile layer as future producer-facing and lower-half-facing contracts.

2. `vc4tile_precision_roadmap_design.md` / `vc4tile_precision_roadmap_design(1).md`  
   Locks the decision that M5 and first end-to-end Triton/IREE drafts are 32-bit-only in executable semantics, while M5 adds strict future-proof precision metadata and deterministic diagnostics for unsupported sub-32 forms.

3. `vc4tile_m5_copy_planner_design.md`  
   Locks the decision that M5 must add a dedicated copy planner early, before tile dot/matmul, because tile movement and layout planning are prerequisites for ergonomic shared transpose, reductions, contractions, and future producer lowering.

This document integrates those three decisions into one M5 milestone plan. Coding agents should treat this document as the high-level implementation contract and the three companion docs as detailed design constraints.

---

## 2. M5 scope and non-scope

### 2.1 What M5 is

M5 is the ergonomic VC4Tile milestone.

The current stack is:

```text
VC4Tile core / ergonomic surface
  -> SSAVC4
  -> scheduled VC4
  -> QASM / runtime artifacts
  -> hardware
```

M5 extends VC4Tile so that kernel authors and future producer lowerings can express tile-level computation and movement in a CuTe/ThunderKittens-like way, while still lowering through the M4/M3/M2 path:

```text
vc4tile ergonomic surface
  -> --canonicalize-vc4tile-surface
  -> --plan-vc4tile-copies
  -> --legalize-vc4tile-core-cfg
  -> --verify-vc4tile-core
  -> --convert-vc4tile-to-ssavc4
  -> --convert-ssavc4-to-vc4
  -> vc4-codegen --emit-bundle
  -> VC4 hardware
```

M5 should make VC4Tile look and feel more like a tile-kernel abstraction while preserving the concrete VC4 constraints:

```text
16 lanes per QPU warp
up to 12 physical QPU/warp slots
one user-visible 4 KiB VPM shared-memory window
TMU global load path
VPM QPU read/write path
VDW global store path
VDR/VCD global-to-VPM load path, to be added early in M5
four-semaphore reusable barrier path
scheduled VC4 sink remains QASM-near
```

### 2.2 What M5 is not

M5 must not implement any producer lowering:

```text
No Triton -> VC4Tile
No IREE -> VC4Tile
No StableHLO -> VC4Tile
No JAX/PyTorch/Torch-MLIR -> VC4Tile
No MLIR gpu -> VC4Tile
No producer -> SSAVC4 shortcut
```

M5 must not implement real sub-32 precision:

```text
No executable f16/bf16/fp8/fp4/int8/int4
No VPM packed/laned sub-32 execution
No regfile pack/unpack semantics exposed as executable paths
No quantization scale/zero-point lowering
No silent widening of producer-like sub-32 requests
```

M5 must not create a direct lower path:

```text
No VC4Tile -> scheduled VC4 pass
No VC4Tile -> QASM path
No fixture-specific path around SSAVC4
```

---

## 3. Required pre-M5 state

Before M5 package generation, the post-M4/pre-M5 SCF/core-CFG staging work should be resolved.

Required passes:

```text
--legalize-vc4tile-core-cfg
--verify-vc4tile-core
```

Required properties:

```text
scf.if and supported scf.for forms lower to cf.br / cf.cond_br and block arguments
raw scf.* is rejected by --convert-vc4tile-to-ssavc4 with an ordering diagnostic
index-typed values do not leak into core
producer dialect ops do not leak into core
zero-trip and non-divisible scf.for semantics are handled correctly or rejected deterministically
substantive SCF fixtures run on real hardware
```

If Stage 3 does not land cleanly, M5-01 must absorb the missing work before ergonomic surface features are added. M5 should not build tile ergonomics on an untrusted SCF legalization.

---

## 4. Global M5 pass pipeline

M5 should establish this canonical full pipeline:

```text
vc4-opt input.mlir \
  --canonicalize-vc4tile-surface \
  --plan-vc4tile-copies \
  --legalize-vc4tile-core-cfg \
  --verify-vc4tile-core \
  --convert-vc4tile-to-ssavc4 \
  -o lowered.ssavc4.mlir

vc4-opt lowered.ssavc4.mlir \
  --convert-ssavc4-to-vc4 \
  -o scheduled.vc4.mlir

vc4-codegen scheduled.vc4.mlir \
  --emit-bundle --output-dir <candidate_dir>
```

The VC4Tile candidate support runner must use this full pipeline for surface fixtures. It must also preserve inspection artifacts:

```text
input.vc4tile.mlir
surface-normalized.vc4tile.mlir      if useful
planned.vc4tile.mlir                 after --plan-vc4tile-copies
core.vc4tile.mlir                    after --legalize-vc4tile-core-cfg
lowered.ssavc4.mlir
scheduled.vc4.mlir
manifest.json / layout.json / kernel_launch.c/h / QASM / shader arrays
```

By default, the runner must regenerate candidates fresh. Reuse may only happen under an explicit opt-in such as:

```text
VC4_REUSE_GENERATED_CANDIDATE=1
```

---

## 5. M5 feature and verification philosophy

Every M5 feature must be implemented vertically:

```text
dialect/type/attr support
invalid diagnostics
surface canonicalization / copy planning / core verification
VC4Tile -> SSAVC4 lowered IR
SSAVC4 -> scheduled VC4 artifact shape when executable
candidate-first generated bundle
real hardware fixture when executable
final regression and anti-shortcut scans
```

Hardware remains the gold standard. If an ergonomic operation is executable, it needs hardware proof. Text-only FileCheck is insufficient for acceptance.

M5 should preserve both:

```text
core fixtures      prove the existing M4/M3/M2 low-level path still works
ergonomic fixtures prove new surface syntax lowers to the same core and works on hardware
```

The implementation should not delete or rewrite existing M4 core fixtures merely because ergonomic versions exist. The old fixtures remain regression protection.

---

## 6. Slice overview

The recommended M5 slice list is:

```text
m5-00-milestone-package
m5-01-surface-core-pipeline-and-runner
m5-02-tile-layout-types-and-precision-markers
m5-03-vdr-vcd-global-to-vpm-load-support
m5-04-copy-planner-v1
m5-05-global-register-tile-load-store
m5-06-shared-vpm-copy-and-transpose
m5-07-scf-composition-with-ergonomic-ops
m5-08-boundary-resource-role-metadata
m5-09-elementwise-tile-compute
m5-10-tile-and-block-reductions
m5-11-tile-contract-dot-matmul
m5-12-final-acceptance
```

The order matters. Movement and layouts come before compute. Reductions and contractions come after copy planning and shared VPM ergonomics. Triton/IREE adapters come after M5, not inside M5.

---

## 7. Slice m5-00: milestone package

### 7.1 Intent

Create the M5 automation package and lock scope.

### 7.2 Expected end state

M5 package files exist and parse. No compiler implementation is required for this slice.

Expected source products:

```text
pro_scripts/milestones/vc4-vc4tile-m5.json
pro_scripts/vc4_vc4tile_m5_worklist.json
pro_scripts/vc4_vc4tile_m5_verifications.json
pro_scripts/vc4_vc4tile_m5_context_profiles.json
pro_scripts/prompts/vc4_vc4tile_m5/README.md
pro_scripts/prompts/vc4_vc4tile_m5/constitution.md
pro_scripts/prompts/vc4_vc4tile_m5/output_contract.md
pro_scripts/prompts/vc4_vc4tile_m5/slice_contract.md
pro_scripts/prompts/vc4_vc4tile_m5/m5_handoff.md
pro_scripts/prompts/vc4_vc4tile_m5/slice00_milestone_package.md
...
pro_scripts/prompts/vc4_vc4tile_m5/slice12_final_acceptance.md
pro_scripts/prompts/vc4_vc4tile_m5/gpt_slice_prompt.md.j2
pro_scripts/prompts/vc4_vc4tile_m5/gpt_failure_prompt.md.j2
pro_scripts/prompts/vc4_vc4tile_m5/gpt_diagnosis_prompt.md.j2
pro_scripts/prompts/vc4_vc4tile_m5/codex_mechanical_prompt.md.j2
pro_scripts/prompts/vc4_vc4tile_m5/codex_contract.md
pro_scripts/prompts/vc4_vc4tile_m5/implementation_integrity_audit_prompt.md
compiler/docs/codegen/vc4tile-m5-copy-planner-design.md
compiler/docs/codegen/vc4tile-m5-compute-primitives-design.md
compiler/docs/codegen/vc4tile-m5-precision-roadmap.md
compiler/docs/codegen/vc4tile-m5-full-design.md
compiler/docs/codegen/vc4tile-m5-plan.md
```

### 7.3 Scope contract

M5 package docs must contain these scope sentences or equivalent:

```text
M5 is ergonomic VC4Tile only.
M5 adds surface tile movement, layouts, copy planning, elementwise tile compute, reductions, and small 32-bit tile contraction/dot/matmul contracts.
M5 does not implement Triton, IREE, StableHLO, JAX, PyTorch, Torch-MLIR, or MLIR gpu lowering.
M5 does not implement executable sub-32 precision.
M5 preserves the vc4tile -> ssavc4 -> scheduled vc4 -> artifact/runtime path.
```

### 7.4 Verifications

Required verifications:

```text
source_products
json_parse
mechanisms_available
milestone_scope_contract
package_audit_contract
regression_contract for current M2/M3/M4 final acceptance, no hardware optional only if explicitly requested
```

---

## 8. Slice m5-01: surface/core pipeline and runner integration

### 8.1 Intent

Add the M5 surface/core boundary and pass-pipeline integration before adding new surface ops.

### 8.2 Required passes

Add/register:

```text
--canonicalize-vc4tile-surface
--plan-vc4tile-copies
```

At this slice, these passes may be mostly no-op except for ordering checks and diagnostics. The slice may introduce `vc4tile.surface_placeholder` as a temporary non-executable surface sentinel. That sentinel exists only to verify the boundary: surface canonicalization erases it, core verification rejects it if it survives, and direct `--convert-vc4tile-to-ssavc4` rejects it with an ordering diagnostic.

Existing/staged passes must be present:

```text
--legalize-vc4tile-core-cfg
--verify-vc4tile-core
--convert-vc4tile-to-ssavc4
```

### 8.3 Required runner behavior

Update `compiler/test/CodeGen/VC4Tile/Support/run_vc4tile_candidate_codegen_test.sh` so ergonomic fixtures use:

```text
input.mlir
  -> surface-normalized.vc4tile.mlir
  -> planned.vc4tile.mlir
  -> core.vc4tile.mlir
  -> lowered.ssavc4.mlir
  -> scheduled.vc4.mlir
  -> bundle
```

The runner must keep M4 core fixtures working. A fixture may opt into `VC4TILE_INPUT_IS_CORE=1` or equivalent if it starts from core input, but M5 ergonomic fixtures should use the full pipeline.

### 8.4 Diagnostics

`--convert-vc4tile-to-ssavc4` must reject unplanned surface ops. The m5-01 direct-rejection test must actually invoke `--convert-vc4tile-to-ssavc4`; a parser-only `vc4-opt file.mlir` command is not a valid rejection test.

```text
error: vc4tile.tile_load is a surface operation; run --canonicalize-vc4tile-surface and --plan-vc4tile-copies before --convert-vc4tile-to-ssavc4
```

### 8.5 Tests

Dialect / conversion tests:

```text
compiler/test/Conversion/VC4TileToSSAVC4/reject-surface-before-canonicalize.mlir
compiler/test/Conversion/VC4TileToSSAVC4/pass-pipeline-empty-surface.mlir
compiler/test/Dialect/VC4Tile/verify-core-reject-surface-placeholder.mlir
```

Codegen/runner tests:

```text
compiler/test/CodeGen/VC4Tile/Emit/pipeline-minimal-surface-vc4tile.mlir
```

### 8.6 Hardware

No new hardware fixture is strictly required if no executable new operation is added. However, the slice should rerun one existing M4 fixture through the updated runner to prove no regression, preferably:

```text
vector_store_smoke_vc4tile
```

### 8.7 Verifications

```text
build-vc4_opt
build-vc4_codegen
build-check_vc4
pass registration command
runner support_script_contract
lowered_ir_contract for no-op pipeline
hardware_cpu_reference_contract on one existing M4 fixture
implementation_integrity_contract
```

---

## 9. Slice m5-02: tile/layout types and precision markers

### 9.1 Intent

Add the tile descriptor vocabulary needed by copy planning and compute primitives.

### 9.2 Required semantic fields

M5 must define a representation for:

```text
logical shape
rank
memory space: global | shared_vpm | register
element type: f32/i32/u32 executable in M5
storage type: f32/i32/u32 executable in M5
accumulator type: f32/i32/u32 executable in M5
layout: row_major | col_major | affine_2d | vpm_row | vpm_col | transposed_view
boundary policy: exact | tail_predicated | zero/clamp if implemented or rejected
role: input | output | accumulator | scratch | metadata-only initially
precision policy: exact_32 only executable
packing: none only executable
```

### 9.3 Suggested syntax

The exact ODS spelling can be refined, but an implementation should aim for something like:

```mlir
#vc4tile.memory_space<global>
#vc4tile.memory_space<shared_vpm>
#vc4tile.memory_space<register>

#vc4tile.layout<row_major>
#vc4tile.layout<col_major>
#vc4tile.layout<affine_2d, strides = [64, 4], offset_unit = byte>
#vc4tile.layout<vpm_row>
#vc4tile.layout<vpm_col>
#vc4tile.layout<transposed_view>

#vc4tile.precision<storage = f32, expressed = f32, accumulator = f32, packing = none>

!vc4tile.tile<1x16xf32, #vc4tile.memory_space<register>, #vc4tile.layout<row_major>>
!vc4tile.tile<16x16xi32, #vc4tile.memory_space<shared_vpm>, #vc4tile.layout<vpm_row>>
```

If MLIR type syntax makes that exact spelling awkward, ODS may store these as op attributes instead. The semantic fields are mandatory either way.

### 9.4 Precision policy

M5 accepts:

```text
f32/i32/u32 storage
f32/i32/u32 expressed type
f32/i32/u32 accumulator type
precision_policy = exact_32
packing = none
```

M5 rejects:

```text
f16, bf16, fp8, fp4, int8, uint8, int4, uint4
packing = vpm_packed/vpm_laned/byte_packed/nibble_packed
quantization scales or zero points
```

Diagnostics must not say “unsupported type” only. They should say:

```text
error: M5 supports only 32-bit executable tile storage; got storage_type = f16
```

### 9.5 Tests

Required tests:

```text
compiler/test/Dialect/VC4Tile/tile-types-layouts-roundtrip.mlir
compiler/test/Dialect/VC4Tile/tile-types-layouts-invalid.mlir
compiler/test/Dialect/VC4Tile/tile-precision-markers-roundtrip.mlir
compiler/test/Dialect/VC4Tile/tile-precision-markers-invalid.mlir
compiler/test/Dialect/VC4Tile/tile-layout-affine-invalid.mlir
```

### 9.6 Hardware

No hardware fixture is required for pure type/attr syntax, but this slice must run `check-vc4` and one existing M4 hardware smoke if the runner changes.

### 9.7 Verifications

```text
source_products
build-vc4_opt
build-check_vc4
dialect_contract
invalid_diagnostic_contract
implementation_integrity_contract
```

---

## 10. Slice m5-03: VDR/VCD global-to-VPM load support

### 10.1 Intent

Add the lower-half global-to-VPM DMA load support needed by the copy planner for regular 2D global-to-shared tile loads.

### 10.2 Required operation

Add an SSAVC4 operation, recommended spelling:

```mlir
ssavc4.vdr.load %base
  { elem_bytes = 4 : i32,
    row_len = 16 : i32,
    nrows = 16 : i32,
    memory_pitch_bytes = 64 : i32,
    vpm_base_row = 0 : i32,
    vpm_base_col = 0 : i32,
    orientation = #ssavc4.vpm_orientation<horizontal>,
    vpitch = 16 : i32,
    serialize = "mutex" }
  : i32
```

The exact name can be `ssavc4.vdr.load`, `ssavc4.vcd.load`, or a project-consistent alternative. It must represent **VCD/VDR DMA load from global memory into VPM**, not a TMU load.

### 10.3 Lowering requirements

Lowering to scheduled VC4 must emit appropriate scheduled pseudoops / bundles for:

```text
VPMVCD_RD_SETUP basic setup
optional extended stride/pitch setup if needed
VPM_LD_ADDR
VPM_LD_WAIT
serialization around VPM/VCD setup if required by existing lower-half policy
```

The implementation must follow existing VDW store conventions for setup/wait serialization.

### 10.4 Tests

Dialect:

```text
compiler/test/Dialect/SSAVC4/vdr-load-roundtrip.mlir
compiler/test/Dialect/SSAVC4/vdr-load-invalid.mlir
```

Conversion:

```text
compiler/test/Conversion/SSAVC4ToVC4/vdr-load.mlir
compiler/test/Conversion/SSAVC4ToVC4/vdr-load-invalid.mlir
```

Emit:

```text
compiler/test/CodeGen/SSAVC4/Emit/vdr-load-ssavc4.mlir
```

Hardware fixture:

```text
compiler/test/CodeGen/SSAVC4/Hardware/Run/vdr_load_roundtrip_ssavc4/input.mlir
compiler/test/CodeGen/SSAVC4/Hardware/Run/vdr_load_roundtrip_ssavc4/expected.json
compiler/test/CodeGen/SSAVC4/Hardware/Run/vdr_load_roundtrip_ssavc4/candidate/vdr_load_roundtrip_ssavc4_candidate_harness.c
```

### 10.5 Hardware oracle

The fixture should:

```text
1. Allocate input and output global buffers.
2. Fill input as a 2D matrix with nontrivial values, e.g. value[row][col] = 1000 + 37*row + col.
3. Fill output and sentinels with poison.
4. Launch kernel.
5. Kernel VDR-loads input tile into VPM.
6. Kernel reads VPM and writes output through existing VDW store or another proven path.
7. Harness copies output back.
8. Harness compares every element and sentinel.
```

### 10.6 Verifications

```text
build-vc4_opt
build-vc4_codegen
build-check_vc4
ssavc4 dialect lit subset
ssavc4 conversion lit subset
ssavc4 emit lit subset
candidate generate/assemble/build/run/expected_json
implementation_integrity_contract
M4 regression slice for shared VPM/VDW remains green
```

---

## 11. Slice m5-04: copy planner v1

### 11.1 Intent

Implement `--plan-vc4tile-copies` and the first set of ergonomic movement ops.

### 11.2 Surface ops

Add:

```text
vc4tile.tile_load
vc4tile.tile_store
vc4tile.copy_tile
vc4tile.tile_view
vc4tile.tile_subview
vc4tile.transpose_view
vc4tile.shared_tile_alloc
```

At this slice, not every op needs all strategies, but every parsed op must either lower correctly or reject deterministically.

### 11.3 Planning strategies

Implement at least:

```text
global -> register       via masked_load_global / TMU lower path
register -> global       via masked_store_global / VPM+VDW lower path
register -> shared_vpm   via shared_store / VPM write
shared_vpm -> register   via shared_load / VPM read
tile_view/subview        folded into source offsets/layout attrs
transpose_view           folded into layout attributes, not materialized unless consumed
```

If VDR landed in m5-03, also implement:

```text
global -> shared_vpm regular 2D via VDR
shared_vpm -> global regular 2D via VDW
```

If VDR did not land, implement a TMU+VPM fallback for at least one global->shared case and keep a required VDR follow-up gate.

### 11.4 Example input

```mlir
%tile = vc4tile.tile_load %in[%zero], %mask
  {shape = [1, 16], layout = #vc4tile.layout<row_major>,
   memory_space = #vc4tile.memory_space<global>, elem_type = i32,
   storage_type = i32, precision = #vc4tile.precision<exact_32>}
  : (i32, i32, vector<16xi1>) -> !vc4tile.tile<1x16xi32, register>
vc4tile.tile_store %tile, %out[%zero], %mask
  {shape = [1, 16], layout = #vc4tile.layout<row_major>,
   memory_space = #vc4tile.memory_space<global>, elem_type = i32,
   storage_type = i32, precision = #vc4tile.precision<exact_32>}
  : !vc4tile.tile<1x16xi32, register>, i32, i32, vector<16xi1>
```

Expected planned core:

```mlir
%lanes = vc4tile.lane_range : vector<16xi32>
%ld = vc4tile.masked_load_global %in, %lanes, %mask ...
vc4tile.masked_store_global %out, %lanes, %ld, %mask ...
```

### 11.5 Tests

Dialect tests:

```text
tile-load-store-roundtrip.mlir
copy-tile-roundtrip.mlir
tile-view-roundtrip.mlir
copy-planner-invalid.mlir
```

Conversion tests:

```text
plan-copy-global-register.mlir
plan-copy-register-global.mlir
plan-copy-register-shared.mlir
plan-copy-shared-register.mlir
plan-transpose-view.mlir
reject-unplanned-copy-before-core.mlir
reject-sub32-copy-m5.mlir
```

Hardware fixtures:

```text
tile_load_store_1d_vc4tile
register_shared_roundtrip_vc4tile
shared_register_roundtrip_vc4tile
```

### 11.6 Verifications

```text
source_products
build-vc4_opt
build-vc4_codegen
build-check_vc4
dialect_contract
invalid_diagnostic_contract
lowered_ir_contract for each planned path
scheduled_artifact_contract for at least tile_load_store_1d
hardware_cpu_reference_contract for each executable fixture
support_script_contract for full pipeline
implementation_integrity_contract
```

---

## 12. Slice m5-05: global/register tile load-store ergonomics

### 12.1 Intent

Prove the ergonomic global/register movement layer with real input-dependent hardware fixtures.

### 12.2 Required behavior

Support:

```text
1D contiguous global -> register tile_load
1D tail-masked tile_load
1D register -> global tile_store
tail-masked tile_store
simple 2D row-major tile load/store by row fragments
simple affine stride if existing TMU/core supports it
```

### 12.3 Example ergonomic SAXPY-style movement

```mlir
%lanes = vc4tile.lane_range : vector<16xi32>
%mask = vc4tile.tail_mask %base_idx, %n : i32, i32 -> vector<16xi1>
%x = vc4tile.tile_load %x_base[%base_idx], %mask
  {shape = [1,16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, elem_type = f32, precision = #vc4tile.precision<exact_32>}
  : (i32, i32, vector<16xi1>) -> !vc4tile.tile<1x16xf32, register>
%y = vc4tile.tile_load %y_base[%base_idx], %mask
  {shape = [1,16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, elem_type = f32, precision = #vc4tile.precision<exact_32>}
  : (i32, i32, vector<16xi1>) -> !vc4tile.tile<1x16xf32, register>
```

The compute part can still use existing core arithmetic or later `tile_fma`.

### 12.4 Hardware fixtures

Required:

```text
tile_load_1d_tail_vc4tile
  Proves tail-masked global loads with poison/sentinel regions.

tile_store_1d_tail_vc4tile
  Proves register->global through VPM/VDW and sentinels.

tile_load_store_2d_row_major_vc4tile
  Proves row-major 2D indexing over multiple rows.

tile_load_store_affine_stride_vc4tile
  Proves a non-unit but supported stride, or deterministic rejection if not supported.
```

Each fixture should validate:

```text
all active elements correct
inactive/tail elements untouched
sentinel regions untouched
checksum or mismatch count derived from copied-back device output
VC4_TEST_RESULT status derived from mismatches/launch failures
```

### 12.5 Verifications

Add fixture matrix:

```json
"m5_copy_global_register": [
  "tile_load_1d_tail_vc4tile",
  "tile_store_1d_tail_vc4tile",
  "tile_load_store_2d_row_major_vc4tile",
  "tile_load_store_affine_stride_vc4tile"
]
```

Run generate/assemble/build/run/expected_json for each.

---

## 13. Slice m5-06: shared VPM copy and transpose

### 13.1 Intent

Turn shared VPM and transpose from mechanical core examples into ergonomic tile programs.

### 13.2 Required behavior

Support:

```text
shared_tile_alloc with static rows/bytes
register -> shared copy
shared -> register copy
global -> shared copy for regular 2D if VDR exists, or fallback if documented
shared -> global copy via VDW
transpose_view over shared tile
shared transpose materialization using VPM orientation
barrier composition where cooperative sharing requires it
```

### 13.3 Example ergonomic shared transpose

```mlir
%src = vc4tile.tile_load %in[0, 0]
  {shape = [16,16], memory_space = #vc4tile.memory_space<global>,
   layout = #vc4tile.layout<row_major>, elem_type = i32,
   precision = #vc4tile.precision<exact_32>}
  : (i32) -> !vc4tile.tile<16x16xi32, global>
%smem = vc4tile.shared_tile_alloc
  {shape = [16,16], elem_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>}
  : !vc4tile.tile<16x16xi32, shared_vpm>
vc4tile.copy_tile %src, %smem
  {src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>}
  : !vc4tile.tile<16x16xi32, global>, !vc4tile.tile<16x16xi32, shared_vpm>
vc4tile.barrier
%t = vc4tile.transpose_view %smem {permutation = [1, 0]}
  : !vc4tile.tile<16x16xi32, shared_vpm> -> !vc4tile.tile<16x16xi32, shared_vpm>
vc4tile.tile_store %t, %out[0, 0]
  {memory_space = #vc4tile.memory_space<global>, layout = #vc4tile.layout<row_major>, elem_type = i32}
  : !vc4tile.tile<16x16xi32, shared_vpm>, i32
```

### 13.4 Hardware fixtures

Required:

```text
shared_tile_roundtrip_vc4tile
  register -> shared -> register -> global

global_to_shared_to_global_2d_vc4tile
  global input staged through shared VPM and stored back

shared_transpose_16x16_ergonomic_vc4tile
  ergonomic replacement for mechanical shared transpose

shared_transpose_store_global_vc4tile
  proves transpose_view can feed tile_store
```

### 13.5 Verifications

```text
dialect_contract for shared tile ops and transpose view
invalid diagnostics for overlarge VPM allocation and unsupported transpose rank
lowered_ir_contract proving surface ops are gone and VPM read/write/VDW/VDR paths appear
scheduled_artifact_contract proving scheduled VPM/VDW/VDR ops appear
hardware_cpu_reference_contract for all fixtures
anti-fake VC4_TEST_RESULT scan
```

---

## 14. Slice m5-07: SCF composition with ergonomic ops

### 14.1 Intent

Prove ergonomic tile ops can appear inside supported `scf.for`/`scf.if` and still lower through the post-M4 core CFG boundary.

### 14.2 Required behavior

Support:

```text
scf.for around tile_load/tile_store/copy_tile
scf.if around tile movement and elementwise ops
iter_args carrying scalar/vector/tile-derived values where legal
zero-trip and non-divisible loop cases handled correctly or rejected according to Stage 3 policy
surface ops planned before SCF is lowered to cf
no scf/index remains before --convert-vc4tile-to-ssavc4
```

### 14.3 Example

```mlir
%zero = arith.constant 0 : index
%ub = arith.constant 4 : index
%step = arith.constant 1 : index
scf.for %i = %zero to %ub step %step {
  %i32 = arith.index_cast %i : index to i32
  %offset = arith.muli %i32, %sixteen : i32
  %mask = vc4tile.mask_all : vector<16xi1>
  %tile = vc4tile.tile_load %in[%offset], %mask ...
  vc4tile.tile_store %tile, %out[%offset], %mask ...
}
```

After canonicalization/planning/legalization:

```text
No scf.for
No index
Loop becomes cf.br/cf.cond_br with block args
Tile ops become core load/store/copy operations
```

### 14.4 Hardware fixtures

Required:

```text
scf_tiled_copy_loop_vc4tile
scf_tiled_copy_zero_trip_vc4tile or deterministic rejection test if zero-trip not executable
scf_tiled_copy_non_divisible_trip_vc4tile or deterministic rejection if not supported
scf_tiled_transpose_loop_vc4tile
scf_tiled_saxpy_loop_vc4tile
```

At least three should run on hardware; rejection-only cases can be invalid diagnostic tests if the semantics are intentionally unsupported.

### 14.5 Verifications

```text
SCF lit subset from post-M4/pre-M5 remains green
new SCF+surface lit subset
core sanitation scan of generated core.vc4tile.mlir: no scf, no index, no surface ops
hardware_cpu_reference_contract for executable fixtures
```

---

## 15. Slice m5-08: boundary, resource, and role metadata

### 15.1 Intent

Add ergonomic metadata that is meaningful for future producer lowering and current verification, even if some fields are not fully consumed yet.

### 15.2 Fields

Add or formalize:

```text
role = input | output | accumulator | scratch | constant | metadata
boundary_policy = exact | tail_predicated | zero | clamp | reject
resource intent = uses_shared_vpm | uses_barrier | vpm_rows | vpm_bytes | semaphores | warps_per_block
copy_stage = none | prologue | steady_state | epilogue, metadata-only in M5 unless consumed
reuse_hint = none | shared | register | producer, metadata-only initially
```

### 15.3 Strictness rule

Metadata-only does not mean decorative. If a field is accepted, it must either:

```text
1. be consumed by verification/planning/resource metadata now, or
2. be preserved deterministically and covered by tests, or
3. be rejected if unsupported.
```

### 15.4 Tests

```text
role-metadata-roundtrip.mlir
role-metadata-invalid.mlir
boundary-policy-roundtrip.mlir
boundary-policy-invalid.mlir
resource-intent-roundtrip.mlir
resource-intent-invalid.mlir
```

### 15.5 Hardware

Hardware is required for any boundary policy that affects execution. For metadata-only role annotations, dialect/lowered-IR tests are enough.

Required executable fixtures if policies are implemented:

```text
boundary_tail_predicated_load_store_vc4tile
boundary_zero_fill_tile_load_vc4tile, only if zero fill is implemented
boundary_clamp_tile_load_vc4tile, only if clamp is implemented
```

If zero/clamp are not implemented, they must be rejected.

---

## 16. Slice m5-09: elementwise tile compute

### 16.1 Intent

Add ergonomic tile compute conveniences before reductions and contractions.

### 16.2 Required ops

Add at least:

```text
vc4tile.tile_fill
vc4tile.tile_broadcast
vc4tile.tile_add
vc4tile.tile_sub
vc4tile.tile_mul
vc4tile.tile_select
```

Optional if lower-half support is clean:

```text
vc4tile.tile_fma
```

### 16.3 Lowering

These are surface ops. They canonicalize to core arithmetic on vector/scalar carriers:

```text
tile_fill      -> arith.constant + splat/load_imm
tile_broadcast -> splat
tile_add       -> arith.addi/addf or existing core ALU op path
tile_mul       -> arith.muli/mulf or existing core ALU op path
tile_select    -> mask/select path if supported, otherwise lower to existing mask logic or reject
```

### 16.4 Example

```mlir
%x = vc4tile.tile_load %x_base[%off], %mask ... -> !vc4tile.tile<1x16xf32, register>
%y = vc4tile.tile_load %y_base[%off], %mask ... -> !vc4tile.tile<1x16xf32, register>
%a = vc4tile.tile_broadcast %alpha {shape = [1,16], elem_type = f32}
%ax = vc4tile.tile_mul %a, %x : !vc4tile.tile<1x16xf32, register>
%out_tile = vc4tile.tile_add %ax, %y : !vc4tile.tile<1x16xf32, register>
vc4tile.tile_store %out_tile, %out[%off], %mask ...
```

### 16.5 Hardware fixtures

```text
tile_elementwise_add_store_vc4tile
tile_mul_store_vc4tile
tile_fma_saxpy_ergonomic_vc4tile, if tile_fma implemented
tile_masked_select_tail_vc4tile
```

### 16.6 Verifications

```text
dialect_contract
invalid_diagnostic_contract
lowered_ir_contract proving surface ops removed
scheduled_artifact_contract proving ALU instructions appear
hardware_cpu_reference_contract for executable fixtures
```

---

## 17. Slice m5-10: tile and block reductions

### 17.1 Intent

Add ergonomic reduction primitives that lower to existing rotate/reduce/barrier/shared-memory mechanisms.

### 17.2 Required ops

Add:

```text
vc4tile.tile_reduce
vc4tile.row_reduce
vc4tile.warp_reduce
vc4tile.block_reduce
```

The exact surface can be smaller if implementation complexity requires, but M5 should include both warp-local and cooperative/block-level reduction examples.

### 17.3 Supported reduction kinds

M5 should start with:

```text
add for i32/u32/f32 if lower-half support exists
min/max only if existing ALU/reduce support makes this straightforward
```

If f32 reduction is not supported by current lower half, it may be deferred, but diagnostics must be explicit.

### 17.4 Example

```mlir
%tile = vc4tile.tile_load %in[%off], %mask ... -> !vc4tile.tile<1x16xi32, register>
%sum = vc4tile.tile_reduce %tile, %mask
  {axis = 1 : i32, kind = #vc4tile.reduce_kind<add>}
  : !vc4tile.tile<1x16xi32, register>, vector<16xi1> -> i32
```

### 17.5 Lowering

```text
warp-local tile_reduce -> existing rotate/reduce core ops
block_reduce -> per-warp partial reduce + shared VPM store + barrier + final warp reduce
```

### 17.6 Hardware fixtures

```text
tile_reduce_sum_i32_vc4tile
warp_reduce_sum_ergonomic_vc4tile
block_reduce_sum_ergonomic_vc4tile
block_reduce_tail_sum_vc4tile
```

### 17.7 Verifications

Must include hardware because reductions are executable and easy to fake with constants.

Harnesses must use input-dependent values and validate exact sums.

---

## 18. Slice m5-11: tile contract, dot, and matmul

### 18.1 Intent

Add the second-half M5 computation contracts that future Triton and IREE/Linalg lowering can target.

This slice implements the compute-primitives decision.

### 18.2 Required ops

Add:

```text
vc4tile.tile_contract
vc4tile.tile_dot
vc4tile.tile_matmul
```

These are ergonomic surface ops. They are not core ops and not SSAVC4 ops.

### 18.3 M5-supported forms

M5 should support small, correctness-first 32-bit forms:

```text
tile_dot:      1x16 dot product, i32 or f32 if lower-half supports it
tile_matmul:   small 4x4 or 8x8 tiled matmul decomposed into tile_dot/elementwise loops
tile_contract: simple contraction over one static K axis
```

M5 should reject:

```text
dynamic ranks
unsupported layouts
sub-32 input/output/accumulation
large tiles that require unimplemented software tiling
implicit global loads inside contract if movement was not expressed
```

### 18.4 Example `tile_dot`

```mlir
%a = vc4tile.tile_load %a_base[%off], %mask ... -> !vc4tile.tile<1x16xf32, register>
%b = vc4tile.tile_load %b_base[%off], %mask ... -> !vc4tile.tile<1x16xf32, register>
%dot = vc4tile.tile_dot %a, %b, %mask
  {accumulator_type = f32, precision = #vc4tile.precision<exact_32>}
  : !vc4tile.tile<1x16xf32, register>, !vc4tile.tile<1x16xf32, register>, vector<16xi1> -> f32
```

### 18.5 Example `tile_contract`

```mlir
%c = vc4tile.tile_contract %a, %b, %acc
  {contracting_dims = [[1], [0]],
   iterator_types = ["parallel", "parallel", "reduction"],
   lhs_layout = #vc4tile.layout<row_major>,
   rhs_layout = #vc4tile.layout<col_major>,
   accumulator_type = f32,
   precision = #vc4tile.precision<exact_32>}
  : !vc4tile.tile<4x16xf32, register>,
    !vc4tile.tile<16x4xf32, shared_vpm>,
    !vc4tile.tile<4x4xf32, register>
    -> !vc4tile.tile<4x4xf32, register>
```

### 18.6 Lowering policy

`tile_contract` / `tile_dot` / `tile_matmul` lower to:

```text
tile_load/copy_tile/tile_store if movement is included in the surface program
or consume already-loaded tile values
then elementwise tile mul/add and tile_reduce/block_reduce
then VC4Tile core arithmetic/reductions
then SSAVC4 ALU/rotate/barrier/shared ops
```

They must not lower directly to SSAVC4 as opaque operations.

### 18.7 Hardware fixtures

```text
tile_dot_1x16_i32_vc4tile
tile_dot_1x16_f32_vc4tile, if f32 arithmetic path is reliable
tile_matmul_4x4_i32_vc4tile
tile_matmul_4x4_f32_vc4tile, if f32 arithmetic path is reliable
tile_contract_shared_rhs_vc4tile
```

At least two substantive fixtures must run on real hardware. If f32 dot/matmul is not ready, use i32 first and leave f32 as a tracked follow-up.

### 18.8 Verifications

```text
dialect_contract
invalid_diagnostic_contract for unsupported precision/layout/rank
lowered_ir_contract proving tile_contract ops are gone after canonicalization
scheduled_artifact_contract proving real ALU/reduce paths
hardware_cpu_reference_contract with CPU matmul/dot oracle
anti-shortcut scan for fixture-name special cases
```

---

## 19. Slice m5-12: final acceptance

### 19.1 Intent

Prove M5 is complete, cumulative, candidate-first, hardware-backed, and scoped correctly.

### 19.2 Required final checks

M5 final acceptance must run:

```text
build-vc4_opt
build-vc4_codegen
build-check_vc4
all M5 targeted dialect/conversion/emit lit subsets
all M5 required hardware fixtures
M4 final acceptance / regression
M3 final/cumulative regression as required by M4 contract
M2 final/cumulative regression as required by M3/M4 contract
implementation integrity audit
copy planner integrity scan
precision scope scan
producer-lowering absence scan
surface/core ordering scan
candidate harness evidence scan
```

### 19.3 Required M5 hardware matrix

Recommended matrix:

```json
"m5_copy_global_register": [
  "tile_load_1d_tail_vc4tile",
  "tile_store_1d_tail_vc4tile",
  "tile_load_store_2d_row_major_vc4tile",
  "tile_load_store_affine_stride_vc4tile"
],
"m5_copy_shared_vpm": [
  "vdr_load_roundtrip_vc4tile",
  "global_to_shared_to_global_2d_vc4tile",
  "register_to_shared_roundtrip_vc4tile",
  "shared_to_register_roundtrip_vc4tile",
  "shared_transpose_16x16_ergonomic_vc4tile",
  "shared_transpose_store_global_vc4tile"
],
"m5_scf_ergonomic": [
  "scf_tiled_copy_loop_vc4tile",
  "scf_tiled_transpose_loop_vc4tile",
  "scf_tiled_saxpy_loop_vc4tile"
],
"m5_elementwise": [
  "tile_elementwise_add_store_vc4tile",
  "tile_mul_store_vc4tile",
  "tile_masked_select_tail_vc4tile"
],
"m5_reductions": [
  "tile_reduce_sum_i32_vc4tile",
  "warp_reduce_sum_ergonomic_vc4tile",
  "block_reduce_sum_ergonomic_vc4tile"
],
"m5_contracts": [
  "tile_dot_1x16_i32_vc4tile",
  "tile_matmul_4x4_i32_vc4tile",
  "tile_contract_shared_rhs_vc4tile"
]
```

This is a large matrix. It may be trimmed for runtime, but every significant feature must have at least one or two real hardware fixtures, and M5 final should include enough cross-product coverage that shortcuts are hard.

### 19.4 Anti-shortcut scans

M5 final acceptance must fail on:

```text
compiler source branching on fixture/public names
direct VC4TileToVC4 pass or command-line option
producer lowering keywords in M5 implementation paths
fake VC4_TEST_RESULT in support runners or candidate harnesses
surface ops reaching --convert-vc4tile-to-ssavc4
sub-32 executable lowering paths beyond metadata/diagnostics
reference bundle edits used as acceptance evidence
stale .vc4_auto reuse without explicit opt-in
hardware fixture outputs computed on host instead of copied back from device
```

### 19.5 Final expected state

After M5 final acceptance:

```text
VC4Tile has a real ergonomic tile surface.
Surface ops canonicalize to VC4Tile core.
Copy planning is hardware-aware and hardware-proven.
Shared transpose can be written ergonomically.
SCF composes with ergonomic tile ops and disappears before core lowering.
Elementwise tile compute works.
At least first reductions work.
At least small 32-bit tile_dot/tile_contract/tile_matmul forms work.
No producer lowering has been added.
No executable sub-32 precision has been added.
M2/M3/M4 regressions remain green.
```

---

## 20. M5 worklist feature gates

The M5 verifier should define feature IDs similar to M4:

```text
vc4tile_surface_core_pipeline
tile_layout_type_model
precision_marker_model
vdr_vcd_load_support
copy_planner_v1
global_register_tile_movement
shared_vpm_tile_movement
transpose_view
scf_ergonomic_composition
boundary_resource_role_metadata
elementwise_tile_compute
tile_reductions
block_reductions
tile_contract
tile_dot
tile_matmul
```

Each feature should declare required verification layers:

```json
{
  "id": "copy_planner_v1",
  "status": "implemented",
  "requires": {
    "dialect": true,
    "invalid_diagnostics": true,
    "lowered_ir": true,
    "scheduled_artifact": true,
    "hardware_cpu_reference": true
  }
}
```

Metadata-only features can mark hardware as false only with a reason:

```json
{
  "id": "role_metadata",
  "status": "implemented_metadata_only",
  "requires": {
    "dialect": true,
    "invalid_diagnostics": true,
    "lowered_ir": true,
    "scheduled_artifact": false,
    "hardware_cpu_reference": false
  },
  "reason": "role metadata is parsed/preserved/rejected but not yet consumed by execution in this slice"
}
```

---

## 21. Code-generation and runner requirements

### 21.1 Runner path

Every M5 ergonomic hardware fixture must use:

```text
compiler/test/CodeGen/VC4Tile/Support/run_vc4tile_candidate_codegen_test.sh <fixture> generate
compiler/test/CodeGen/VC4Tile/Support/run_vc4tile_candidate_codegen_test.sh <fixture> assemble
compiler/test/CodeGen/VC4Tile/Support/run_vc4tile_candidate_codegen_test.sh <fixture> build
compiler/test/CodeGen/VC4Tile/Support/run_vc4tile_candidate_codegen_test.sh <fixture> run
```

The runner must lower through the full M5 pipeline.

### 21.2 Candidate harnesses

Every harness must:

```text
include generated kernel_launch.h
call the generated <fixture>_launch wrapper or vc4LaunchKernel path produced by the generated wrapper
allocate/copy device buffers through the runtime APIs
copy device output back
compare against CPU oracle
check sentinel regions
print VC4_TEST_RESULT with status determined by real mismatches and launch failures
```

Allowed diagnostic fields:

```text
mismatches
total_mismatches
sentinel_mismatches
checksum_expected
checksum_actual
runtime_launches read from vc4ProgramLaunches(program), not hard-coded
launch_failures read from vc4ProgramLaunchFailures(program), not hard-coded
```

Forbidden:

```text
fixed status=PASS
host-computed output substituted for device output
hard-coded runtime_launches=1 unless read from runtime
reference run.sh acceptance path
fixture-name branch inside compiler
```

---

## 22. Expected M5 files by category

This is not exhaustive, but package generation should include source-product expectations in these areas.

### 22.1 Dialect/type definitions

```text
compiler/include/vc4/Dialect/VC4Tile/IR/VC4TileTypes.td
compiler/include/vc4/Dialect/VC4Tile/IR/VC4TileAttrs.td
compiler/include/vc4/Dialect/VC4Tile/IR/VC4TileOps.td
compiler/include/vc4/Dialect/VC4Tile/IR/VC4TileTypes.h
compiler/lib/Dialect/VC4Tile/IR/VC4TileTypes.cpp
compiler/lib/Dialect/VC4Tile/IR/VC4TileOps.cpp
```

### 22.2 Conversion/passes

```text
compiler/include/vc4/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.h
compiler/lib/Conversion/VC4TileToSSAVC4/VC4TileToSSAVC4.cpp
compiler/lib/Conversion/VC4TileToSSAVC4/VC4TileSurfaceCanonicalize.cpp, optional if split
compiler/lib/Conversion/VC4TileToSSAVC4/VC4TileCopyPlanner.cpp, optional if split
compiler/tools/vc4-opt/vc4-opt.cpp
```

### 22.3 SSAVC4 VDR support

```text
compiler/include/vc4/Dialect/SSAVC4/IR/SSAVC4Ops.td
compiler/lib/Dialect/SSAVC4/IR/SSAVC4Ops.cpp
compiler/lib/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.cpp
compiler/test/Dialect/SSAVC4/vdr-load-roundtrip.mlir
compiler/test/Conversion/SSAVC4ToVC4/vdr-load.mlir
compiler/test/CodeGen/SSAVC4/Hardware/Run/vdr_load_roundtrip_ssavc4/...
```

### 22.4 VC4Tile M5 tests

```text
compiler/test/Dialect/VC4Tile/*tile*.mlir
compiler/test/Conversion/VC4TileToSSAVC4/*tile*.mlir
compiler/test/CodeGen/VC4Tile/Emit/*tile*.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/*_vc4tile/input.mlir
compiler/test/CodeGen/VC4Tile/Hardware/Run/*_vc4tile/expected.json
compiler/test/CodeGen/VC4Tile/Hardware/Run/*_vc4tile/candidate/*_candidate_harness.c
```

---

## 23. Concrete final pipeline examples

### 23.1 Lit command for surface-to-core boundary

```mlir
// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core -o - | FileCheck %s --check-prefix=CORE
// CORE-NOT: vc4tile.tile_load
// CORE-NOT: vc4tile.tile_store
// CORE-NOT: vc4tile.copy_tile
// CORE-NOT: scf.
// CORE-NOT: index
// CORE: vc4tile.masked_load_global
// CORE: vc4tile.masked_store_global
```

### 23.2 Lit command for rejecting raw surface before conversion

```mlir
// RUN: not vc4-opt %s --convert-vc4tile-to-ssavc4 2>&1 | FileCheck %s --check-prefix=ERR
// ERR: surface operation
// ERR: --plan-vc4tile-copies
```

### 23.3 Hardware validation snippet

```sh
VC4_HW_ATTEMPT_TIMEOUT_SEC=60 \
VC4_HARDWARE_TIMEOUT_SEC=60 \
VC4_RUN_SH_MAX_ATTEMPTS=3 \
bash compiler/test/CodeGen/VC4Tile/Support/run_vc4tile_candidate_codegen_test.sh \
  shared_transpose_16x16_ergonomic_vc4tile run
```

Expected log evidence:

```text
VC4_KERNEL_LAUNCH name=shared_transpose_16x16_ergonomic_vc4tile ...
VC4_TEST_RESULT name=shared_transpose_16x16_ergonomic_vc4tile status=PASS total_mismatches=0 sentinel_mismatches=0 checksum_actual=... checksum_expected=...
```

---

## 24. Things that look fishy or need deliberate handling

### 24.1 VDR is missing today

The copy planner naturally wants VDR/VCD for regular global-to-shared loads. If M5 tries to skip VDR entirely, global-to-shared ergonomic copies will either be inefficient TMU+VPM loops or under-specified. The right move is to add VDR early as M5-03.

### 24.2 M5 scope is large

The proposed M5 includes copy planning, shared VPM ergonomics, SCF composition, elementwise compute, reductions, and small contractions. This is intentionally comprehensive, but implementation may be long. The verifier package should be slice-strict so partial work cannot masquerade as full M5.

If runtime becomes a concern, final acceptance can group hardware fixtures into focused matrices, but it should not drop coverage for entire feature families.

### 24.3 `tile_contract` may reveal lower-half f32 gaps

If f32 arithmetic/reduction support is weaker than i32, M5 should implement i32 dot/matmul first and reject f32 contraction deterministically until proven. Do not fake f32 correctness or rely on constant outputs.

### 24.4 Metadata-only fields can become dead weight

Role/resource/precision fields must be tested and either consumed, preserved, or rejected. Decorative fields that are ignored by every pass should be avoided.

### 24.5 Surface/core split must stay strict

M5 should not let surface ops leak into `--convert-vc4tile-to-ssavc4`. The strict ordering is what lets future Triton/IREE lowering target a stable ergonomic surface while preserving a low-level core accepted by SSAVC4 lowering.

---

## 25. Final M5 decision statement

M5 will make VC4Tile a real ergonomic tile-kernel layer while preserving the existing lower-half architecture. It will add tile/layout/precision metadata, a hardware-aware copy planner, ergonomic global/shared/register tile movement, shared VPM transpose, SCF composition, elementwise tile compute, reductions, and small 32-bit tile contraction/dot/matmul contracts. It will remain 32-bit-only in executable semantics and will not implement producer lowering. Every executable feature will have real hardware proof through fresh candidate-generated artifacts, device copyback, CPU oracles, sentinels, and cumulative M2/M3/M4 regressions.

After M5, the next milestones can lower Triton and then IREE/JAX/PyTorch into VC4Tile rather than into SSAVC4 or scheduled VC4 directly. Later precision milestones can fill in sub-32 storage, packing, quantization, and specialized low-precision compute using the metadata and copy-planner hooks added here.
