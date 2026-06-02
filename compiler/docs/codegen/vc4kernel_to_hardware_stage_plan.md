# VC4Kernel to SSAVC4 to Hardware Roadmap

**Status:** implementation roadmap after corrected `vc4kernel` design decisions.  
**Date:** 2026-05-31.  
**Current baseline:** repo was clean at `256234e66ade441b60bf624fb5c521ebaea350e6` after pre-lowering cleanups; VC4Tile was excised; `vc4kernel` source and conversion lit were runnable from source; `check-vc4` was green.  
**Scope:** get from the current pre-lowering state to a corrected, locked `vc4kernel` surface that lowers through SSAVC4/scheduled VC4/codegen/runtime to real VC4 hardware, and is then ready for the upstream standard `vector` surface and eventual Triton TTIR ingestion.

---

## 0. Executive summary

The project is not blocked by design ambiguity anymore. The remaining work is a sequence of deliberate implementation slices:

```text
P0  Rewrite spec/docs to corrected surface.
P1  Update vc4kernel ODS/parser/verifier/tests to corrected surface.
P2  Refactor SSAVC4/VC4 resource metadata and launch ABI around semantic resources.
P3  Update runtime/manifest planner for vpm_base_row/semaphore_base residency assignment.
P4  Implement scalar f32 uniforms and scalar i1 condition infrastructure.
P5  Implement predicate/mask infrastructure, fragment_cmp, pred.any/all, fragment_select, cf.cond_br.
P6  Implement hardware-derived VPM/VDR/VDW mode schema and lower-half emission.
P7  Implement VPM allocator with explicit user rows and hidden compiler staging rows.
P8  Implement/fix vc4kernel -> ssavc4 lowering for launch identity and fragment compute.
P9  Implement VDW register->global store through hidden VPM staging and hardware-test it first.
P10 Implement masked TMU loads and hardware-test zero-fill/no-OOB semantics.
P11 Implement explicit VPM/VDR/VDW paths, horizontal and vertical 32-bit, with hardware tests.
P12 Implement full i32 multiply semantics with mul24 fast path and software fallback.
P13 Implement cooperative barrier using semaphore_base and hardware-test it.
P14 Run final Stage 1 acceptance audit.
P14.5 Dynamic rectangular transfer / runtime GEMM-GEMV checkpoint.
P15 Only after P14.5, start the upstream standard vector surface specification.
```

The plan intentionally updates the lower half where needed. Existing SSAVC4/scheduled-VC4 compatibility is not a design constraint.

---

## 1. Current baseline and facts to preserve

From the final pre-lowering context collection:

```text
HEAD: 256234e66ade441b60bf624fb5c521ebaea350e6
Commit: Excise retired VC4Tile source tree
Worktree after tests: clean
VC4Tile tracked paths/content hits: 0
Direct VC4KernelToVC4 tracked paths/content hits: 0
FunctionOpInterface present: yes
Source VC4Kernel dialect lit: 42/42 passed
Source VC4KernelToSSAVC4 lit: 9/9 passed
Build-tree dialect lit: 42/42 passed
Build-tree conversion lit: 9/9 passed
check-vc4: 292/292 passed
```

The current surface lock is valid relative to the old strict spec, but the old strict spec has now been corrected. Therefore the next step is not lowering yet; the next step is a controlled spec/surface correction, followed by lower-half refactors and hardware-proven lowering.

Non-negotiable invariants from this point forward:

```text
- no compatibility/alias phase
- no VC4Tile resurrection
- no direct vc4kernel -> scheduled vc4 path
- no producer/vector/Triton lowering until vc4kernel -> hardware is accepted
- no fixture/public_name special cases
- no host-side result substitution
- no stale reference QASM or stale generated bundles as acceptance evidence
- no executable sub-32 precision
```

---

## 2. Roadmap structure and verification philosophy

Each implementation slice must have:

```text
source-product check:
  exact files changed intentionally

build check:
  ninja -C compiler/build vc4-opt
  ninja -C compiler/build vc4-codegen when codegen/runtime touched

dialect/conversion tests:
  source-root lit where applicable
  build-tree or check-vc4 as cumulative fallback

semantic checks:
  FileCheck assertions on emitted IR, not just parse success

integrity checks:
  grep/static scans for direct paths, old ops, shortcut strings, fixture-name dispatch

hardware checks:
  required for every new executable memory/predicate/barrier path
```

Every Codex prompt should retain the established response policy:

```text
SUCCESS only if all requested verification passed.
FAILURE only if truly blocked, ambiguous, or success would require cheating/shortcuts/weakened checks.
If a mechanical/test harness issue is honestly fixable within prompt scope, fix it and rerun.
Do not commit merely because success was reported; commit previous work only when the next prompt clearly indicates moving on.
```

---

## P0 — Correct the strict specification and planning docs

### Goal

Replace the old strict spec with the corrected final contract.

### Required edits

Update:

```text
compiler/docs/codegen/vc4kernel_dialect_strict_specification.md
```

and any nearby planning docs that still contradict the corrected spec.

Required spec changes:

```text
- remove vc4kernel.lane_id
- remove vc4kernel.block_id
- remove user-authored resource dictionary
- keep vc4kernel.program_id, vc4kernel.warp_id, vc4kernel.lane_range
- define warps_per_block as a kernel source attr
- define compiler-computed vc4.resource lower-half metadata
- define runtime builtins program_id, warp_id, vpm_base_row, semaphore_base, optional warps_per_block
- keep fragment_cmp and define it as producing general_mask unless proven structured
- broaden !vc4kernel.pred<16> to full/empty/tail_prefix/rect_row/general_mask
- define consumer-specific predicate support/fallbacks
- define vdw_store_fragment as planning op through hidden VPM staging
- define user and hidden compiler VPM allocation rows
- define hardware-derived VPM/VDR/VDW mode schema
- require horizontal and vertical 32-bit VPM/VDR/VDW support in v1
- keep sub-32/packed/laned schema but reject executable use
```

### Tests / verification

Documentation-only slice:

```bash
set -euo pipefail
python3 - <<'PY'
from pathlib import Path
p = Path('compiler/docs/codegen/vc4kernel_dialect_strict_specification.md')
s = p.read_text()
for forbidden in ['vc4kernel.lane_id', 'vc4kernel.block_id', 'resource = {']:
    if forbidden in s and 'Explicitly removed' not in s:
        raise SystemExit(f'forbidden old surface text may still be normative: {forbidden}')
for required in ['vpm_base_row', 'semaphore_base', 'general_mask', 'compiler_vpm_staging_rows_per_warp', 'vc4kernel.fragment_cmp']:
    assert required in s, required
print('spec correction doc check passed')
PY
ninja -C compiler/build check-vc4
git diff --check
```

### Acceptance

The docs clearly state that the old surface lock is superseded by the corrected spec, and no document tells future agents to target `lane_id`, `block_id`, old source `resource`, old `vc4tile`, or normalizable-only predicates.

---

## P1 — Update VC4Kernel ODS, parser/printer, verifier, and surface tests

### Goal

Make the actual `vc4kernel` dialect match the corrected spec.

### Required source changes

Likely files:

```text
compiler/include/vc4/Dialect/VC4Kernel/IR/VC4KernelOps.td
compiler/include/vc4/Dialect/VC4Kernel/IR/VC4KernelAttrs.td
compiler/include/vc4/Dialect/VC4Kernel/IR/VC4KernelTypes.td
compiler/lib/Dialect/VC4Kernel/IR/VC4KernelOps.cpp
compiler/lib/Dialect/VC4Kernel/IR/VC4KernelAttrs.cpp
compiler/lib/Dialect/VC4Kernel/IR/VC4KernelTypes.cpp
compiler/lib/Conversion/VC4KernelToSSAVC4/VC4KernelToSSAVC4.cpp, only for verifier helpers that currently live there
compiler/test/Dialect/VC4Kernel/*.mlir
```

Implement exactly:

```text
1. Delete BlockIdOp and LaneIdOp from ODS and C++.
2. Keep ProgramIdOp, WarpIdOp, LaneRangeOp.
3. Replace kernel source resource dict requirement with warps_per_block attr.
4. Keep public_name, schedule_mode, arg_attrs, function_type.
5. Add precise ODS attrs for vpm_orientation<horizontal|vertical>, vpm_width<w32|w16|w8>, vpm_subword<none|packed|laned>.
6. Update vpm_read_fragment/vpm_write_fragment signatures to include y and x operands and mode attrs.
7. Add or update vdw_store_vpm op and keep/update vdw_store_vpm_fragment if needed by fragment path.
8. Change predicate verifier from normalizable-only to class-aware with general_mask support.
9. Revise consumer legality: fragment_select/pred.any/pred.all/tmu/vpm accept general_mask; vdr rejects predicates; vdw general masks require fallback support in later lowering or verifier accepted only if conversion supports it.
10. Remove invalid-fragment-cmp-predicate-consumer or rewrite it into a consumer-specific unsupported fallback diagnostic if such a consumer exists.
11. Add invalid tests for forbidden block_id and lane_id.
12. Add invalid test for user-authored resource dict.
13. Add invalid tests for executable subword VPM modes.
```

### Surface test updates

Update roundtrip tests:

```text
kernel-roundtrip.mlir:
  no resource dict, has warps_per_block.

identity-roundtrip.mlir:
  program_id, warp_id where legal, lane_range; no block_id/lane_id.

predicate-roundtrip.mlir:
  includes fragment_cmp/general_mask or separate general-mask test.

vpm-roundtrip.mlir:
  horizontal and vertical 32-bit modes with y/x.

vdr-vpm-roundtrip.mlir:
  horizontal and vertical VDR/VPM cases.

resource-roundtrip.mlir:
  becomes resource-computation or removed from source surface; source should not carry computed resource dict.
```

### Verification

```bash
set -euo pipefail
ninja -C compiler/build vc4-opt
run_lit() {
  if command -v llvm-lit >/dev/null 2>&1; then llvm-lit "$@";
  elif [ -x /opt/homebrew/opt/llvm/bin/llvm-lit ]; then /opt/homebrew/opt/llvm/bin/llvm-lit "$@";
  else python3 -m lit "$@"; fi
}
run_lit -sv compiler/test/Dialect/VC4Kernel
run_lit -sv compiler/test/Conversion/VC4KernelToSSAVC4
ninja -C compiler/build check-vc4
git diff --check
python3 - <<'PY'
from pathlib import Path
paths = [Path('compiler/include/vc4/Dialect/VC4Kernel'), Path('compiler/lib/Dialect/VC4Kernel'), Path('compiler/lib/Conversion/VC4KernelToSSAVC4'), Path('compiler/test/Dialect/VC4Kernel')]
text = '\n'.join(p.read_text(errors='replace') for base in paths for p in base.rglob('*') if p.is_file() and p.suffix in {'.td','.cpp','.h','.mlir'})
assert 'vc4kernel.block_id' not in text or 'invalid-forbidden-block-id' in text
assert 'vc4kernel.lane_id' not in text or 'invalid-forbidden-lane-id' in text
assert 'vc4kernel.thread_id' not in text or 'invalid-forbidden-thread-id' in text
for s in ['vpm_base_row','general_mask','warps_per_block']:
    assert s in text, s
print('P1 static check passed')
PY
```

### Acceptance

The ODS op inventory now matches the corrected spec; source `vc4kernel` no longer contains `block_id`, `lane_id`, or a user-authored resource dict; focused dialect/conversion tests and `check-vc4` are green.

---

## P2 — Refactor SSAVC4 / scheduled VC4 resource metadata

### Goal

Replace compatibility resource fields with the semantic computed resource model.

### Required source changes

Likely files:

```text
compiler/include/vc4/Dialect/SSAVC4/IR/SSAVC4Ops.td
compiler/lib/Dialect/SSAVC4/IR/SSAVC4Ops.cpp
compiler/include/vc4/Dialect/VC4/IR/VC4Ops.td
compiler/lib/Dialect/VC4/IR/VC4Ops.cpp
compiler/lib/Conversion/SSAVC4ToVC4/SSAVC4ToVC4.cpp
compiler/tools/vc4-codegen/*
compiler/test/Dialect/SSAVC4/*
compiler/test/Conversion/SSAVC4ToVC4/*
compiler/test/CodeGen/SSAVC4/*
compiler/test/CodeGen/VC4/*
```

Semantic metadata fields to support:

```text
schedule_mode
warps_per_block
qpu_slots_per_block
uses_tmu
uses_vpm
uses_vpm_qpu_read
uses_vpm_qpu_write
uses_vdr
uses_vdw
uses_barrier
user_vpm_rows_per_block
compiler_vpm_staging_rows_per_warp
compiler_vpm_staging_rows_per_block
total_vpm_rows_per_block
total_vpm_bytes_per_block
semaphore_count_per_block
requires_vpm_base_row_builtin
requires_semaphore_base_builtin
```

Delete or stop relying on:

```text
shared_vpm_bytes
user_shared_vpm_rows_per_block
uses_shared_vpm
source-authored require_full_block_residency
source-authored semaphores_per_block
```

If an old field remains temporarily for internal compatibility, it must be derived and marked deprecated, not source-authored or semantically primary.

### Tests

Add/update SSAVC4 lit tests:

```text
semantic-resource-roundtrip.mlir
invalid-resource-over-vpm-rows.mlir
invalid-resource-over-semaphores.mlir
resource-no-vpm-no-builtins.mlir
resource-vpm-base-row-required.mlir
resource-barrier-semaphore-base-required.mlir
```

### Verification

```bash
set -euo pipefail
ninja -C compiler/build vc4-opt
ninja -C compiler/build vc4-codegen
run_lit -sv compiler/test/Dialect/SSAVC4
run_lit -sv compiler/test/Conversion/SSAVC4ToVC4
ninja -C compiler/build check-vc4
git diff --check
git grep -n "shared_vpm_bytes\|user_shared_vpm_rows_per_block\|uses_shared_vpm" -- compiler/include/vc4 compiler/lib compiler/test || true
```

Expected grep result: no normative uses; if temporary derived uses remain, they must be explicitly documented and not required by `vc4kernel`.

### Acceptance

Lower half accepts the semantic resource model and rejects invalid resource usage. Existing lower-half tests remain green.

---

## P3 — Runtime/manifest launch-resource planner

### Goal

Make `vc4-codegen`/runtime assign `vpm_base_row` and `semaphore_base` according to computed resources.

### Required source changes

Likely files:

```text
compiler/tools/vc4-codegen/*
external/libpi/*
compiler/test/CodeGen/VC4/*
compiler/test/CodeGen/SSAVC4/*
vc4_test_specs/*
```

Implement:

```text
resident_blocks = min(
  floor(12 / warps_per_block),
  total_vpm_rows_per_block == 0 ? infinity : floor(64 / total_vpm_rows_per_block),
  semaphore_count_per_block == 0 ? infinity : floor(16 / semaphore_count_per_block)
)

vpm_base_row(slot) = slot * total_vpm_rows_per_block
semaphore_base(slot) = slot * semaphore_count_per_block
```

Uniform packing:

```text
formal args first in source order
then runtime builtins in deterministic ABI order:
  program_id, warp_id, vpm_base_row, semaphore_base, warps_per_block if required
```

Independent-vector:

```text
warps_per_block = 1
each resident logical request consumes one block slot
vpm_base_row varies by resident request if VPM/staging used
semaphore_base omitted
```

Cooperative-block:

```text
each logical block consumes warps_per_block QPU launches
all warps in same block get same program_id, vpm_base_row, semaphore_base
warp_id differs 0..warps_per_block-1
```

### Tests

Add manifest/codegen tests:

```text
resource_manifest_no_vpm.mlir
resource_manifest_independent_vpm_base_rows.mlir
resource_manifest_cooperative_block_vpm_and_semaphores.mlir
uniform_packing_builtin_order.mlir
```

### Verification

```bash
set -euo pipefail
ninja -C compiler/build vc4-codegen
ninja -C compiler/build check-vc4
git diff --check
```

### Acceptance

Generated manifests and host launchers expose the computed residency and pack builtin uniforms deterministically, without using physical QPU number for logical VPM ownership.

---

## P4 — Scalar f32 uniform and scalar i1 condition infrastructure

### Goal

Support true scalar f32 formals and a clean scalar condition model in SSAVC4/lower-half.

### Required implementation

```text
1. Ensure ssavc4.uniform.read supports i32 and f32 scalar result types.
2. Ensure ssavc4.splat supports scalar f32 -> vector<16xf32>.
3. Represent scalar i1 results from arith.cmpi, pred.any, pred.all as condition plans in VC4KernelToSSAVC4 lowering.
4. Add or harden SSAVC4 condition ops as needed:
   - make_flags
   - cond_select
   - cond_br
5. Do not treat scalar i1 as a persistent hardware flag SSA value. Flags are mutable machine state and must be materialized near consumers.
```

### Tests

```text
compiler/test/Conversion/VC4KernelToSSAVC4/formal-args-launch-abi.mlir
compiler/test/Conversion/VC4KernelToSSAVC4/scalar-f32-uniform-splat.mlir
compiler/test/Conversion/VC4KernelToSSAVC4/scalar-i1-cmpi-select-branch.mlir
compiler/test/Dialect/SSAVC4/scalar-f32-uniform-read.mlir
compiler/test/Conversion/SSAVC4ToVC4/scalar-f32-uniform-read.mlir
```

### Verification

```bash
set -euo pipefail
ninja -C compiler/build vc4-opt
run_lit -sv compiler/test/Conversion/VC4KernelToSSAVC4
run_lit -sv compiler/test/Dialect/SSAVC4
run_lit -sv compiler/test/Conversion/SSAVC4ToVC4
ninja -C compiler/build check-vc4
git diff --check
```

### Acceptance

An f32 formal can be read from uniform stream, splatted, and used in f32 fragment arithmetic. Scalar i1 control can lower to branch/select without pretending flags are persistent SSA values.

---

## P5 — Predicate/mask infrastructure

### Goal

Implement corrected predicate semantics.

### Required implementation

Add a lowering state structure in `VC4KernelToSSAVC4.cpp` or split into helper files:

```cpp
enum class PredKind {
  Full,
  Empty,
  TailPrefix,
  RectRow,
  GeneralMask
};

struct PredicatePlan {
  PredKind kind;
  Value maskValue;        // for materialized general masks if needed
  Value base;
  Value limit;
  Value row;
  Value rows;
  Value colBase;
  Value cols;
  Value activeCount;      // optional scalar active count for structured masks
};

struct ConditionPlan {
  enum Kind { ConstantFalse, ConstantTrue, ScalarCompare, PredicateAny, PredicateAll } kind;
  ...
};
```

Implement lowering for:

```text
pred.full / empty / tail / rect
pred.and / or / not
fragment_cmp -> general_mask
pred.any / pred.all -> ConditionPlan
fragment_select -> SSAVC4 conditional select
cf.cond_br -> SSAVC4 conditional branch
```

Required behavior:

```text
- full/empty canonicalized when easy.
- tail/rect compute active count when needed.
- general_mask remains legal for fragment_select, pred.any/all, TMU, VPM, VDW fallback paths.
- no vector<16xi1> appears.
```

### Tests

```text
predicate-tail-lowering.mlir
predicate-general-mask-lowering.mlir
fragment-cmp-select-lowering.mlir
control-flow-block-args.mlir
pred-any-all-branch-lowering.mlir
```

### Verification

```bash
set -euo pipefail
ninja -C compiler/build vc4-opt
run_lit -sv compiler/test/Dialect/VC4Kernel
run_lit -sv compiler/test/Conversion/VC4KernelToSSAVC4
ninja -C compiler/build check-vc4
git diff --check
git grep -n "vector<16xi1>" -- compiler/include/vc4/Dialect/VC4Kernel compiler/lib/Dialect/VC4Kernel compiler/lib/Conversion/VC4KernelToSSAVC4 compiler/test/Dialect/VC4Kernel compiler/test/Conversion/VC4KernelToSSAVC4 || true
```

Only invalid tests should mention `vector<16xi1>`.

### Acceptance

General masks are represented and lowered honestly; old non-normalizable-predicate rejection no longer blocks legitimate `fragment_cmp`/Triton-style masks.

---

## P6 — Hardware-derived VPM/VDR/VDW mode schema

### Goal

Update SSAVC4/scheduled VC4/QASM emission to support horizontal/vertical/strided 32-bit VPM/VDR/VDW modes and schema for future sub-32.

### Required implementation

Add lower-half mode attrs/enums:

```text
orientation: horizontal | vertical
width: w32 | w16 | w8
subword_mode: none | packed | laned
x/y coordinates
stride for VPM QPU read/write setup
row_len / nrows / memory_pitch_bytes / vpm_pitch for VDR
units/depth / memory_stride_bytes / block_mode for VDW
```

Executable v1 restrictions:

```text
width = w32 only
subword_mode = none only
orientation = horizontal or vertical
stride/pitch/blockmode supported where hardware supports them
```

Update scheduled vc4 and qasm emitter to encode the real hardware setup bits, not old string names.

### Tests

```text
compiler/test/Dialect/SSAVC4/vpm-mode-roundtrip.mlir
compiler/test/Dialect/SSAVC4/vdr-vdw-mode-roundtrip.mlir
compiler/test/Conversion/SSAVC4ToVC4/vpm-horizontal-vertical-setup.mlir
compiler/test/Conversion/SSAVC4ToVC4/vdr-vdw-horizontal-vertical-setup.mlir
compiler/test/Conversion/SSAVC4ToVC4/reject-subword-executable.mlir
compiler/test/CodeGen/SSAVC4/vpm-vdr-vdw-qasm-modes.mlir
```

### Verification

```bash
set -euo pipefail
ninja -C compiler/build vc4-opt
ninja -C compiler/build vc4-codegen
run_lit -sv compiler/test/Dialect/SSAVC4
run_lit -sv compiler/test/Conversion/SSAVC4ToVC4
run_lit -sv compiler/test/CodeGen/SSAVC4
ninja -C compiler/build check-vc4
git diff --check
```

### Acceptance

Lower half can represent and emit 32-bit horizontal and vertical VPM/VDR/VDW paths using hardware-derived mode attrs. Sub-32 schema exists but executable use is rejected.

---

## P7 — VPM allocator with explicit user rows and hidden staging rows

### Goal

Implement planned row allocation relative to runtime `vpm_base_row`.

### Required implementation

Build a `VPMAllocationPlan` in VC4KernelToSSAVC4:

```cpp
struct VPMAllocationPlan {
  int userRowsPerBlock;
  int stagingRowsPerWarp;
  int stagingRowsPerBlock;
  int totalRowsPerBlock;
  DenseMap<Value, TileAlloc> explicitTiles;
};

struct TileAlloc {
  int baseRowOffset;
  int rows;
  int elemBytes;
};
```

Assignment:

```text
explicit vpm_alloc rows:
  assigned from row offset 0 upward in source order

per-warp hidden staging:
  starts at user_vpm_rows_per_block
  physical row = vpm_base_row + user_vpm_rows_per_block + warp_id * staging_rows_per_warp + local_staging_row

block-level hidden staging:
  after per-warp staging rows
```

All VPM/VDR/VDW paths use:

```text
physical_y = vpm_base_row + allocation_base_row + local_y
```

### Tests

```text
vpm-allocation-user-rows.mlir
vpm-allocation-hidden-staging.mlir
vpm-allocation-multiple-tiles.mlir
vpm-allocation-overflow-reject.mlir
vpm-base-row-builtin-lowering.mlir
```

### Verification

```bash
set -euo pipefail
ninja -C compiler/build vc4-opt
run_lit -sv compiler/test/Conversion/VC4KernelToSSAVC4
ninja -C compiler/build check-vc4
git diff --check
```

### Acceptance

All VPM accesses are relative to `vpm_base_row`; explicit and hidden rows are non-overlapping and resource metadata reflects total rows.

---

## P8 — Launch identity and fragment compute lowering

### Goal

Complete core non-memory lowering.

### Required implementation

```text
- formal i32/f32 args -> uniform.read
- program_id -> builtin uniform
- warp_id -> builtin uniform where legal/needed
- lane_range -> element_number vector
- splat i32/f32
- fragment_add/sub/shl
- fragment_mul f32 -> fmul
- fragment_rotate
- fragment_reduce via rotate/ALU tree
- vc4kernel.return -> ssavc4.thread_end
- cf.br/cf.cond_br with block args
```

Keep i32 multiply fallback for P12 if too large for this slice, but do not silently use mul24 for general i32. Until P12, either reject i32 fragment_mul deterministically or route through a temporary semantic `ssavc4.imul32` that P12 implements.

### Tests

```text
formal-args-launch-abi.mlir
identity-lowering.mlir
lane-range-lowering.mlir
fragment-arith-lowering.mlir
fragment-rotate-lowering.mlir
fragment-reduce-lowering.mlir
control-flow-block-args.mlir
```

### Verification

```bash
set -euo pipefail
ninja -C compiler/build vc4-opt
run_lit -sv compiler/test/Conversion/VC4KernelToSSAVC4
ninja -C compiler/build check-vc4
git diff --check
```

### Acceptance

Straight-line fragment computation and scalar control lower cleanly to SSAVC4 with no direct scheduled VC4 ops.

---

## P9 — First hardware path: VDW register-to-global store through hidden VPM staging

### Goal

Establish the first real `vc4kernel -> hardware` path after corrected surface/lower-half changes.

### Required implementation

For `vdw_store_fragment` full predicate:

```text
1. verify contiguous row offsets
2. compute global base + base_byte_offset
3. allocate hidden per-warp staging row
4. write register fragment to VPM staging row
5. issue VDW store from staging row to global memory
6. wait for store completion as required
```

For tail predicate:

```text
- preserve inactive destination lanes
- either use active-prefix VDW if proven safe, or use fallback RMW path
```

For general mask:

```text
1. TMU load old destination row
2. merged = fragment_select(mask, new_value, old_value)
3. write merged to hidden staging row
4. VDW store full row
```

### Hardware fixtures

```text
vc4kernel_vector_store_full
vc4kernel_vector_store_tail_preserve
vc4kernel_vector_store_general_mask_preserve
```

Each fixture must initialize output to sentinel values and prove inactive lanes preserve the sentinel.

### Verification

```bash
set -euo pipefail
ninja -C compiler/build vc4-opt
ninja -C compiler/build vc4-codegen
run_lit -sv compiler/test/Conversion/VC4KernelToSSAVC4
ninja -C compiler/build check-vc4
# plus fixture runner commands that generate fresh candidates, assemble/build/run on Pi, and CPU-reference compare
git diff --check
```

### Acceptance

Device writes are from actual VC4 execution. No direct register-to-global fiction remains in the lower half.

---

## P10 — Masked TMU loads

### Goal

Implement `tmu_load_fragment` for full, empty, tail, and general predicates.

### Required implementation

```text
full:
  issue TMU request/read for all lanes.

empty:
  issue no request; return zero vector.

tail/general:
  compute safe_offsets = select(pred, requested_offsets, safe_offset)
  if any(pred): issue TMU request/read using safe_offsets
  result = select(pred, raw, zero)
```

The safe address must be a valid in-bounds address for the fixture/program path. For tail loops, lane 0 active address is usually safe when any lane is active; empty path skips the request.

### Hardware fixtures

```text
vc4kernel_tmu_load_full
vc4kernel_tmu_load_tail_zero_fill
vc4kernel_tmu_load_general_mask_zero_fill
vc4kernel_saxpy_f32
```

Test `n=0,1,15,16,17,31` where applicable.

### Acceptance

No inactive lane causes an out-of-bounds TMU request; inactive results are zero-filled.

---

## P11 — Explicit VPM/VDR/VDW paths

### Goal

Implement explicit shared/VPM movement paths with horizontal and vertical 32-bit support.

### Required implementation

```text
vpm_write_fragment:
  register -> VPM, inactive lanes zero, horizontal/vertical w32

vpm_read_fragment:
  VPM -> register, inactive lanes zero, horizontal/vertical w32

vdr_load_to_vpm:
  global rectangle -> VPM, full rectangular DMA, horizontal/vertical w32, memory_pitch_bytes, vpm_pitch

vdw_store_vpm:
  VPM rectangle -> global, full rectangular DMA, horizontal/vertical w32, memory_stride_bytes, block_mode

vdw_store_vpm_fragment:
  VPM fragment -> global with predicate and preserve-destination semantics
```

### Hardware fixtures

```text
vc4kernel_vpm_roundtrip_horizontal_32
vc4kernel_vpm_roundtrip_vertical_32
vc4kernel_vdr_to_vpm_horizontal_32
vc4kernel_vdr_to_vpm_vertical_32
vc4kernel_vpm_to_global_vdw_horizontal_32
vc4kernel_vpm_to_global_vdw_vertical_32
```

### Acceptance

All horizontal and vertical 32-bit VPM/VDR/VDW modes specified in the corrected v1 schema have actual executable hardware proof.

---

## P12 — Full i32 multiplication

### Goal

Implement full 32-bit modular integer multiplication with a `mul24` fast path.

### Required implementation

```text
- add semantic ssavc4.imul32 or equivalent pseudo-op
- implement range proof for 24-bit-safe operands
- lower 24-bit-safe operands to hardware mul24
- lower general operands to software 32-bit multiply fallback
- ensure fragment_mul i32 never silently truncates to mul24 semantics
```

Initial range proofs may be conservative:

```text
- constants in [0, 2^24)
- lane_range and small affine offsets proven within range
- future range attrs from vector/Triton lowering
```

### Hardware fixtures

```text
vc4kernel_imul32_fast_mul24
vc4kernel_imul32_software_fallback
```

Use values that distinguish full 32-bit multiply from mul24 truncation.

### Acceptance

The fallback path produces CPU-reference-correct full 32-bit results.

---

## P13 — Cooperative barrier

### Goal

Implement `vc4kernel.barrier` through semantic resource metadata, runtime `semaphore_base`, and SSAVC4 barrier/semaphore sequence.

### Required implementation

```text
- barrier legal only in cooperative_block
- warps_per_block in [1,12]
- resource computes semaphore_count_per_block=4
- runtime assigns semaphore_base per resident cooperative block
- all warps in block share program_id, vpm_base_row, semaphore_base
- warp_id differs per warp
- barrier sequence uses semaphore_base + 0..3
```

### Hardware fixture

```text
vc4kernel_qpu_barrier_syncthreads
```

The fixture should prove cross-warp synchronization and shared VPM visibility.

### Acceptance

Multiple QPU warps in one cooperative block synchronize correctly, and no fixed global semaphore IDs are hard-coded as the semantic model.

---

## P14 — Full Stage 1 acceptance audit

### Required commands

```bash
set -euo pipefail
ninja -C compiler/build vc4-opt
ninja -C compiler/build vc4-codegen
run_lit -sv compiler/test/Dialect/VC4Kernel
run_lit -sv compiler/test/Conversion/VC4KernelToSSAVC4
run_lit -sv compiler/test/Dialect/SSAVC4
run_lit -sv compiler/test/Conversion/SSAVC4ToVC4
run_lit -sv compiler/test/CodeGen/VC4Kernel
ninja -C compiler/build check-vc4
git diff --check

git grep -n "vc4tile\|VC4Tile\|VC4TileToSSAVC4" -- compiler pro_scripts docs README.md CMakeLists.txt && exit 1 || true
git grep -n "VC4KernelToVC4\|convert-vc4kernel-to-vc4\|vc4kernel-to-vc4" -- compiler pro_scripts && exit 1 || true
git grep -n "vc4kernel.thread_id\|vc4kernel.block_id\|vc4kernel.lane_id" -- compiler/include/vc4/Dialect/VC4Kernel compiler/lib/Dialect/VC4Kernel compiler/lib/Conversion/VC4KernelToSSAVC4 && exit 1 || true
git grep -n "public_name.*==\|fixture\|expected.json\|VC4_TEST_RESULT" -- compiler/lib/Conversion/VC4KernelToSSAVC4 && exit 1 || true
```

### Required hardware suite

```text
vc4kernel_vector_store_full
vc4kernel_vector_store_tail_preserve
vc4kernel_vector_store_general_mask_preserve
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

### Acceptance

`vc4kernel` Stage 1 is accepted only if every required source, lit, conversion, scheduled artifact, codegen, runtime, hardware, and integrity gate passes from fresh generation.

---

## Dynamic rectangular transfer / runtime GEMM-GEMV checkpoint

This checkpoint is required before vector-surface lock if blocked GEMM/GEMV and future `vector.contract` are expected to use VPM reuse with runtime M/N/K. It implements dynamic VDR/VDW rectangular movement, 3D logical launch identity, and hardware fixtures for blocked runtime GEMM/GEMV.

Acceptance requires:

```text
- 3D program_id(axis) and num_programs(axis) lowering and hardware proof.
- Dynamic VDR rect load zero-fill hardware proof.
- Dynamic VDW rect store preserve hardware proof.
- Runtime-pitch/stride variants including non-multiple dimensions.
- Row-by-row fallback when pitch/stride cannot be encoded as one hardware setup word.
- Naive and blocked GEMV with runtime m,n.
- Naive and blocked GEMM with runtime m,n,k.
- Natural row-major contiguous layouts using runtime leading dimensions, not padded fixture-specific strides.
```

The blocked GEMM/GEMV fixtures must use compile-time tile/block maximum sizes but runtime problem sizes and leading dimensions. Static rectangular VDR/VDW ops remain legal for fixed-shape fixtures and fully specialized kernels, but they are not the permanent representation for runtime M/N/K shared-memory blocking.

---

## P15 — Begin upstream standard vector surface only after P14

After P14 and the dynamic rectangular transfer / runtime GEMM-GEMV checkpoint, begin a new design/implementation stage for the standard vector surface.

Initial vector surface should likely be:

```text
func.func or lightweight wrapper with vc4.kernel metadata
memref<?xi32/#vc4.global>, memref<?xf32/#vc4.global>
vector<16xi32>, vector<16xf32>
vector.step
vector.transfer_read/write
vector.mask / vector.create_mask
arith vector ops
scf/cf, with scf lowered before verified vc4kernel
```

First value-layer kernels:

```text
handwritten vector add i32
handwritten SAXPY f32
handwritten tail cases n=0/1/17/31
multi-request program_id case
```

Do not start real TTIR ingestion until handwritten vector -> vc4kernel -> hardware is proven.

---

## Codex prompt sequencing recommendation

The roadmap is large enough that prompts should be narrow and commit after each success. Recommended prompt packages:

```text
01_spec_correction_docs
02_vc4kernel_remove_lane_block_resource_surface
03_vc4kernel_predicate_general_mask_surface
04_vc4kernel_vpm_mode_surface
05_ssavc4_resource_schema
06_runtime_resource_planner
07_scalar_f32_uniform_i1_condition
08_predicate_condition_lowering
09_vpm_vdr_vdw_mode_lower_half
10_vpm_allocator_hidden_staging
11_launch_identity_fragment_compute
12_vdw_store_hardware_full_tail_general
13_tmu_masked_loads_hardware
14_explicit_vpm_vdr_vdw_hardware
15_imul32_hardware
16_barrier_hardware
17_final_acceptance_audit
```

Each prompt must include exact files, exact tests, exact commands, expected output, and SUCCESS/FAILURE contract.

---

## Highest-risk engineering areas

```text
1. General masked VDW store preserve semantics:
   Requires old-row load + merge + staging + store without races.

2. Runtime vpm_base_row assignment:
   Must be consistent across independent-vector and cooperative-block launch modes.

3. Barrier semaphore_base:
   Must avoid global fixed semaphores if more than one cooperative block can be resident.

4. Full i32 multiplication:
   Software fallback must be correct and not accidentally optimized into mul24-only semantics.

5. VPM vertical addressing:
   Must respect hardware y/x alignment and addressing bit constraints.

6. Condition/flag lifetime:
   VC4 flags are machine state, not stable SSA values. The lowering must recompute/materialize near consumers.

7. Hardware freshness:
   Every accepted fixture must regenerate from vc4kernel input; stale bundles invalidate acceptance.
```

---

## Final checkpoint before vector work

Before moving upward, the answer must be:

```text
VC4KERNEL_SURFACE_LOCKED_CORRECTED=YES
VC4KERNEL_TO_SSAVC4_COMPLETE=YES
SSAVC4_LOWER_HALF_SEMANTIC_RESOURCE_MODEL=YES
VC4KERNEL_HARDWARE_FIXTURES_PASS=YES
READY_FOR_STANDARD_VECTOR_SURFACE=YES
READY_FOR_TRITON_TTIR=NO, not until vector surface is separately locked and hardware-proven
```
