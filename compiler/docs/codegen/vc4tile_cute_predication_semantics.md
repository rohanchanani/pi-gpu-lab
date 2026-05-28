# VC4Tile CuTe-Idiomatic Predication Semantics, Constrained by VC4 Hardware

**Intended repo path:** `compiler/docs/codegen/vc4tile-cute-predication-semantics.md`  
**Status:** normative design contract for the post-M5 predication hardening work  
**Scope:** VC4Tile predicate semantics, predicate normalization targets, inactive-lane policies, fragment-planning obligations, VC4 hardware legality, and verification expectations  
**Non-scope:** Triton lowering, IREE/JAX/PyTorch lowering, executable sub-32 precision, new scheduled-VC4 shortcuts, and any direct VC4Tile-to-VC4 lowering path

---

## 1. Purpose

This document locks the target semantics for making VC4Tile predication CuTe-idiomatic while remaining honest about VideoCore IV hardware constraints.

Later implementation steps should treat this document as the semantic source of truth. The goal is to prevent ad hoc per-fixture predicate matching and to make later patches mostly mechanical:

```text
define predicate model
  -> normalize predicate algebra
  -> plan fragments
  -> verify legality
  -> map to VC4 hardware mechanisms
  -> attach to every memory/compute consumer
  -> test on real hardware
```

The central decision is:

```text
VC4Tile predicates are CuTe-style logical-coordinate predicates.
They normalize into VC4-hardware-legal transfer/compute fragments.
They are not merely vector<16xi1> lane masks, and they are not layout-specific bit hacks.
```

---

## 2. Evidence and design sources

This document is grounded in these sources and project decisions:

### 2.1 Real CuTe predication

CuTe's predication model keeps the tile shape fixed, constructs coordinate/identity tensors, partitions those tensors in the same way as the data tensors, and uses predicate tensors congruent with the copied tensors. CuTe's `copy_if` takes a predication tensor with the same shape as the source/destination tensor and copies only the elements whose predicate entries are nonzero.

References:

- NVIDIA CUTLASS CuTe predication: <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0y_predication.html>
- NVIDIA CUTLASS CuTe algorithms / `copy_if`: <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/04_algorithms.html>

### 2.2 VC4 hardware facts

The VC4 QPU programming model is not a general arbitrary mask-tensor engine. It is a structured 16-lane SIMD machine with VPM/VCD/VDW movement modes.

The VC4 architecture guide describes QPUs as effectively 16-way 32-bit SIMD processors, VPM as a 2D array visible to QPUs, and VPM/VCD/VDW as structured mechanisms for QPU/VPM/global movement. The practical VC4-as-CUDA mapping guide locks the backend's execution model as one tiny CUDA-like SM with twelve physical warp slots, sixteen lanes per warp, and one global 4 KiB user-visible VPM shared-memory window.

References:

- `VideoCoreIV-AG100-R Architecture Reference Guide`
- `compiler/docs/codegen/vc4-as-cuda-mapping-guide.md`

### 2.3 M5 compiler architecture

M5 establishes ergonomic VC4Tile surface operations that canonicalize through the VC4Tile pipeline before lowering to SSAVC4:

```text
vc4tile ergonomic surface
  -> --canonicalize-vc4tile-surface
  -> --plan-vc4tile-copies
  -> --legalize-vc4tile-core-cfg
  -> --verify-vc4tile-core
  -> --convert-vc4tile-to-ssavc4
  -> --convert-ssavc4-to-vc4
  -> vc4-codegen --emit-bundle
  -> hardware
```

Predicate semantics must fit this path. No semantic predicate algebra may survive into SSAVC4; it must be normalized and planned into concrete VC4Tile core fragments, masks, control flow, and memory atoms before SSAVC4 conversion.

References:

- `compiler/docs/codegen/vc4tile-m5-full-design.md`
- `compiler/docs/codegen/vc4tile-m5-copy-planner-design.md`
- `compiler/docs/codegen/vc4tile-m5-compute-primitives-design.md`

### 2.4 Precision policy

M5 executable semantics are 32-bit only. Predicate semantics are boolean/logical, but executable predicated transfer/compute in this phase is limited to 32-bit carriers. Sub-32 predicate-aware transfer/compute is future work and must diagnose deterministically.

Reference:

- `compiler/docs/codegen/vc4tile-precision-roadmap-design.md`

---

## 3. Normative definitions

### 3.1 Logical tile coordinate

A **logical tile coordinate** is an element coordinate in the semantic tile/view being consumed.

Examples:

```text
1D tile: coord = lane_or_element
2D tile: coord = (row, col)
contraction K dimension: coord = k
```

Logical coordinates are independent of physical transfer carriers. A logical 8x8 tile may later decompose into four 4x4 fragments, eight 1x8 row fragments, or other legal fragments, but the predicate's meaning is always over logical coordinates.

### 3.2 Predicate

A **predicate** is a boolean function over a logical coordinate domain:

```text
P(coord) -> bool
P(row, col) -> bool
P(k) -> bool
```

The primary VC4Tile predicate model is not `vector<16xi1>`. A vector mask is only one lowered carrier for a fragment. Semantic predicates must retain rank, shape, coordinate meaning, layout/view association, and inactive behavior.

### 3.3 Predicate/data congruence

A predicate is **congruent** with a tile consumer when its coordinate domain exactly matches the logical coordinate domain expected by that consumer.

Examples:

```text
vc4tile.tile_load A_view with pred_A
vc4tile.tile_store C_view with pred_C
vc4tile.tile_contract A, B -> C with pred_A, pred_B, pred_C
```

A predicate for `A_view` cannot silently be reused for `B_view`, a transposed view, or an output view unless the compiler can prove the coordinate transform and explicitly rebase the predicate.

### 3.4 Fragment

A **fragment** is a planned, hardware-mappable subset of a predicate's active coordinate set.

The target fragment classes are:

```text
empty
full_tile
row_fragment(row, start_col, width)
row_set_fragment(row_start, row_count, start_col, width)
column_fragment(col, start_row, height)          // legal only where hardware/layout support exists
scalar_fragment(row, col)
sparse_fragment_set([...])                       // explicit fallback only
```

Fragments are planning artifacts. They should not change the semantic tile shape or predicate meaning.

### 3.5 Dense predicate

A **dense fragmentable predicate** is a predicate that can be represented as one or more full, rectangular, row interval, column interval, or tail fragments without arbitrary scattered element selection.

Examples:

```text
full tile
empty tile
rectangular bounds
row-major 1D tail
2D M/N tail
K-tail for contraction input
```

### 3.6 Sparse predicate

A **sparse predicate** is a predicate that cannot be represented without arbitrary scattered element selection.

Sparse lowering is not the main path. It is an explicit fallback class. A sparse predicate must either:

```text
1. lower through an explicit sparse fallback allowed by the consumer/path, or
2. be rejected with a deterministic diagnostic.
```

No implementation may accidentally turn every hard predicate into per-lane guarded stores and call that the normal tile path.

---

## 4. Core semantic decisions

### Decision 1: predicates are logical coordinate predicates, not raw lane masks

**Normative rule:** A VC4Tile predicate is interpreted over the logical coordinate domain of the tile/view/consumer. `vector<16xi1>` is a carrier only after normalization/planning.

**Reason:** This matches CuTe's identity/coordinate tensor model and keeps predicate semantics stable across views, tile shapes, and fragment decomposition.

**Implementation consequence:** Predicate operations must carry enough shape/rank/view information to know what logical coordinates they describe.

---

### Decision 2: tile extents stay static; predicates handle tails

**Normative rule:** Runtime tails do not shrink the semantic tile shape. A tile has its scheduled static shape, and predicates mark invalid logical coordinates inactive.

Examples:

```text
valid:  8x8 tile with predicate active only on rows < M_tail and cols < N_tail
invalid semantic model: runtime 5x7 tile replacing the scheduled 8x8 tile
```

**Reason:** CuTe keeps tiled iteration uniform and handles imperfect divisibility by predicates. This is also friendlier to VC4's fixed 16-lane and 4x4/1x16 physical carriers.

**Implementation consequence:** All tile types/layouts remain statically shaped in the surface/core pipeline; dynamic bounds become predicate operands.

---

### Decision 3: every predicated consumer must declare predicate congruence

**Normative rule:** A predicated consumer must make the predicate/view association explicit. The compiler must reject ambiguous predicate use.

Examples:

```text
load(A_view, pred_for_A_view)                 // valid
store(C_view, pred_for_C_view)                // valid
store(transpose(C_view), pred_for_C_view)     // invalid unless predicate is explicitly transformed
```

**Reason:** Predicate truth is over logical coordinates; views can change coordinate mapping.

**Implementation consequence:** Verifiers and planners must know the consumer view, layout, rank, and shape when interpreting predicates.

---

### Decision 4: loads zero-fill inactive coordinates and do not read inactive memory

**Normative rule:** A predicated load has this semantics:

```text
if P(coord):
    result(coord) = memory(coord)
else:
    result(coord) = fill_value
```

For M5/M6 v1, `fill_value` defaults to zero for numeric 32-bit tile loads.

**Required safety:** Inactive coordinates must not perform out-of-bounds global/shared reads.

**Reason:** K-tail and bounds predicates must prevent invalid memory reads and must not feed stale values into reductions or contractions.

**Implementation consequence:** Hardware fixtures must check zero-fill for inactive load lanes/elements whenever the inactive result is observable.

---

### Decision 5: stores preserve inactive destination coordinates and do not write inactive memory

**Normative rule:** A predicated store has this semantics:

```text
if P(coord):
    memory(coord) = value(coord)
else:
    memory(coord) is preserved
```

**Required safety:** Inactive coordinates must not perform out-of-bounds writes and must not clobber sentinel data.

**Reason:** Tail stores must not overwrite memory outside the logical problem domain.

**Implementation consequence:** Every predicated store hardware fixture must include sentinel/guard regions and verify inactive destination preservation.

---

### Decision 6: shared-memory inactive behavior is explicit

**Normative rule:** A predicated copy into shared VPM must declare one inactive destination policy:

```text
zero_fill
preserve
```

Defaults by semantic role:

```text
global/register/shared load for compute input:      zero_fill
K-tail staging into shared for contraction input:   zero_fill
store-like update into an existing shared tile:     preserve unless explicitly zero_fill
shared -> global store path:                        preserve inactive global destination
register -> global store path:                      preserve inactive global destination
```

**Reason:** VC4 has one global 4 KiB user-visible VPM shared-memory window. Stale VPM data is a real correctness hazard, especially when a later contraction reads staged K-tail data.

**Implementation consequence:** Shared-copy ops must not rely on implicit old VPM contents. They must carry or infer a clear inactive policy.

---

### Decision 7: dense rectangular/tail predicates are first-class

**Normative rule:** These predicates must be first-class dense/fragmentable classes:

```text
full tile
empty tile
1D contiguous tail interval
2D rectangular bounds
row interval
row set with common contiguous column interval
K-tail input predicate
M/N output-tail predicate
```

**Reason:** These are the common CuTe/Triton tiled-kernel masks and are naturally mappable to VC4 row/rectangle fragments.

**Implementation consequence:** These forms must be represented directly by the model/planner rather than discovered by repeated ad hoc matcher code in every memory path.

---

### Decision 8: arbitrary sparse masks are explicit fallback or deterministic rejection

**Normative rule:** If a predicate cannot be normalized into dense fragments, it is sparse. Sparse predicates may lower only through an explicit sparse fallback, and only for consumers/paths that opt into that fallback.

**Sparse fallback semantics:** Sparse fallback is correct but slow. It may use guarded scalar or short fragments, but it must be classified as fallback in the plan and diagnostics.

**Reason:** VC4 provides structured transfer modes and 16-lane vectors, not arbitrary dynamic 2D predicated block movement.

**Implementation consequence:** No generic path may silently turn every unrecognized predicate into per-lane stores. The plan must say `sparse_fallback`, or the compiler must reject.

---

### Decision 9: predicate algebra is semantic and normalized

**Normative rule:** Predicate algebra is over logical coordinate functions:

```text
P && Q
P || Q
!P
true
false
```

Canonical simplifications:

```text
P && true  = P
P || false = P
P && false = empty
P || true  = full
!!P        = P
P && P     = P
P || P     = P
```

Supported dense compositions should normalize to finite fragments. Unsupported compositions should lower to explicit sparse fallback if allowed, otherwise reject.

**Reason:** Real CuTe predicate use composes coordinate bounds, tails, and tile/view partitions. Pattern-specific matching of only one spelling is not robust enough.

**Implementation consequence:** The planner should receive normalized predicate expressions/fragments, not arbitrary raw operation names.

---

### Decision 10: layout affects address mapping and fragment splitting, not predicate truth

**Normative rule:** Predicate truth is layout-independent. A predicate is evaluated in logical coordinates. Layout determines how those logical coordinates map to memory addresses, VPM rows/columns, and physical fragments.

Example:

```text
P(row, col) over a transposed logical view still means logical row/col of that view.
The planner may split it differently, but it cannot reinterpret row as col silently.
```

**Reason:** CuTe predicates come from logical coordinate tensors. Layout transformations and tiling preserve coordinate meaning while changing physical mapping.

**Implementation consequence:** Layout-specific planners must explicitly transform/rebase predicates when changing views.

---

### Decision 11: larger logical tiles decompose into supported physical carriers

**Normative rule:** A semantic tile may be larger than a physical fragment. It must decompose into legal physical fragments without changing semantics.

Supported first physical carriers:

```text
1x16 row/vector fragment
4x4 matrix fragment
VPM row fragment
VPM rectangular/static block fragment where legal
scalar fragment only for explicit sparse fallback
```

**Reason:** VC4 has 16-lane SIMD and structured VPM/VDR/VDW modes. M5's tile contracts and copy planner already use 4x4 and 1x16-style carriers.

**Implementation consequence:** A logical 8x8/16x16 tile should be represented as a planner-level decomposition, not by inventing a fake 64-lane or 256-lane hardware mask.

---

### Decision 12: VC4 hardware legality is part of predicate semantics

**Normative rule:** The planner must never claim a predicate is supported if lowering would require unavailable VC4 behavior.

Locked legality facts:

```text
- 16 SIMD lanes per QPU warp.
- User-visible VPM is 64 rows x 16 32-bit words = 4 KiB.
- VPM/VCD/VDW expose structured horizontal/vertical/2D access modes.
- Dynamic active-lane behavior is a 1D row/tail mechanism unless proven otherwise.
- Arbitrary dynamic 2D masks are not a first-class hardware transfer mode.
- Dynamic VPM alignment must not be faked; split into static-alignment cases or reject.
```

**Implementation consequence:** Unsupported hardware combinations must diagnose. They must not generate approximate or best-effort invalid lowering.

---

### Decision 13: fast paths and fallback paths are explicit

**Normative rule:** The lowering hierarchy is:

```text
full static rectangular fragment:
  use block VDR/VDW/VPM mechanisms where legal

static row fragment:
  use row VPM/VDW/VDR or normal vector path

dynamic row tail:
  use one-row dynamic active-lane lowering where already proven

row skip:
  use branch/guarded control flow

sparse predicate:
  use explicit sparse fallback only if allowed

dynamic arbitrary 2D predicate:
  reject unless normalized into supported fragments
```

**Implementation consequence:** Planner output should identify path class, e.g. `full_block`, `row_tail`, `guarded_row_skip`, `sparse_fallback`, so tests can verify that dense cases do not accidentally use sparse fallback.

---

### Decision 14: dynamic predicates are allowed only when statically classifiable

**Normative rule:** Dynamic scalar bounds are allowed if the predicate class remains statically known.

Legal examples:

```text
lane < n
row < rows
col < cols
(row < rows) && (col < cols)
k < K_tail
```

Current required dynamic operand type:

```text
scalar i32
```

Reject cases requiring:

```text
arbitrary data-dependent sparse layouts
dynamic VPM alignment that cannot be statically split
dynamic 2D active-lane block movement not proven on hardware
```

---

### Decision 15: predicate constructors are surface-level semantic operations

**Normative rule:** The surface predicate vocabulary should include these constructors, even if exact op names differ:

```text
mask_all(shape)
mask_none(shape)
tile_bounds_mask(tile_origin, logical_shape, tile_shape)
tail_mask(base, limit, width)
rect_mask(row_start, row_count, col_start, col_count, tile_shape)
logical_and(P, Q)
logical_or(P, Q)
logical_not(P)
```

Each constructor must define:

```text
rank
shape
coordinate domain
dynamic operands
fragmentability
sparse-fallback eligibility
```

**Implementation consequence:** Avoid undocumented one-off mask values attached to consumers. Predicate meaning must be recoverable by the normalizer.

---

### Decision 16: predicates must not survive past the intended boundary

**Normative rule:** Predicate lifetime is:

```text
VC4Tile surface:
  symbolic predicates are allowed.

Canonicalization / copy planning:
  predicates normalize into fragment plans.

VC4Tile core:
  only concrete masks/fragments/control flow/core memory atoms remain.

SSAVC4:
  no semantic predicate algebra remains.
```

**Implementation consequence:** `--convert-vc4tile-to-ssavc4` must reject raw surface predicates or unplanned predicate/copy operations with an ordering diagnostic.

---

### Decision 17: resource use is part of fragment-plan semantics

**Normative rule:** A predicate fragment plan must account for resource effects:

```text
VPM rows used
VPM bytes used
temporary register/vector carriers
barriers required
branch/fragment count for sparse fallback
whether full-block residency assumptions change
```

**Reason:** VC4's shared VPM and barrier resources are globally constrained, and predicate planning may introduce extra staging or branches.

**Implementation consequence:** Resource metadata and verifier checks must be updated when fragment planning changes shared/VPM or barrier use.

---

### Decision 18: predicate semantics are precision-independent, but executable paths are 32-bit-only now

**Normative rule:** Predicates are boolean/logical. Executable predicated transfer/compute in M5/M6 v1 is 32-bit-only.

Supported now:

```text
i32/u32/f32 scalar-like values
vector<16xi32>
vector<16xf32>
32-bit global/shared/register tile movement
32-bit reductions/contracts/matmul paths
```

Unsupported now:

```text
f16/bf16/fp8/int8/uint8/int4/uint4 executable predicated movement or compute
packed/nibble predicated VPM lowering
quantized predicate-aware storage/compute
```

**Implementation consequence:** Sub-32 predicated paths must diagnose; they must not silently widen or pack.

---

## 5. Consumer-specific semantics

### 5.1 `tile_load`

```text
result(coord) = memory(coord) if P(coord)
result(coord) = zero          if !P(coord)
```

Requirements:

```text
- Inactive memory must not be read.
- Result shape remains the tile shape.
- Predicate rank/shape must match the load view.
- Inactive fill defaults to zero for M5/M6 v1.
```

### 5.2 `tile_store`

```text
memory(coord) = value(coord) if P(coord)
memory(coord) is preserved   if !P(coord)
```

Requirements:

```text
- Inactive memory must not be written.
- Sentinel preservation is mandatory in hardware fixtures.
- Predicate rank/shape must match the store view.
```

### 5.3 `copy_tile`

`copy_tile` semantics depend on path and destination inactive policy:

```text
global -> register:
  inactive source not read; destination register zero-filled

shared -> register:
  inactive source not semantically read; destination register zero-filled

global -> shared:
  inactive source not read; shared destination zero-filled by default for compute-input staging

register -> shared:
  inactive shared destination preserved by default unless zero_fill is explicitly requested

shared -> global:
  inactive global destination preserved

register -> global:
  inactive global destination preserved
```

### 5.4 `tile_select`

```text
result(coord) = P(coord) ? true_value(coord) : false_value(coord)
```

The predicate must be congruent with both value operands or must be explicitly rebased.

### 5.5 Reductions

For add/sum-like reductions:

```text
inactive coords contribute additive identity zero
```

For max/min-like reductions:

```text
inactive coords contribute the correct identity or are skipped;
all-empty behavior must be explicitly defined or rejected.
```

All reductions must define empty predicate behavior.

### 5.6 `tile_dot`, `tile_contract`, and `tile_matmul`

Input K-tail predicates:

```text
inactive K coordinates contribute zero
```

Output M/N-tail predicates:

```text
inactive output coordinates preserve destination on store
```

Accumulator rule:

```text
no stale accumulator value may leak into inactive output stores
```

Matrix predicate rule:

```text
matrix predicates must be per-output-coordinate predicates;
scalar dot/broadcast conventions must not be used for matrix tails.
```

---

## 6. Verifier obligations

### 6.1 Before planning

Verify:

```text
predicate rank matches consumer rank
predicate tile shape matches consumer tile shape
predicate/view/layout association is explicit
dynamic operands are scalar i32 where required
inactive destination policy is declared or defaultable by semantic role
sub-32 executable paths are rejected
sparse fallback eligibility is explicit
```

### 6.2 After planning

Verify:

```text
no semantic predicate algebra remains past planning
no unplanned surface predicate/copy ops remain before SSAVC4 conversion
all fragments are hardware-legal for their memory path/layout
dynamic active-lane use is one-row only unless separately proven
dynamic VPM alignment is not required or is statically split
VPM rows/bytes and barrier resources are in bounds
unsupported layout/predicate combinations diagnose rather than approximate
```

---

## 7. Diagnostic requirements

Diagnostics should explain the semantic reason, not only that something is unsupported.

Examples:

```text
unsupported predicate: cannot normalize (P || Q) into dense row fragments and sparse fallback is disabled

unsupported layout/predicate: transposed affine view would require dynamic VPM column alignment

unsupported store predicate: inactive-lane preservation is required but this path only supports full-row overwrite

unsupported load predicate: inactive zero-fill requested but destination shared policy is preserve

unsupported sparse fallback: consumer tile_contract does not permit sparse fallback for K-tail input
```

---

## 8. Hardware proof contract

Every executable predicate class must eventually have real hardware proof.

Hardware is also required during development whenever a patch changes executable
lowering behavior, planner decisions that select executable lowering paths, VC4
hardware mapping, runtime-visible scheduling/resource behavior, or the expected
behavior of an existing or new hardware fixture. This requirement applies even
when the change is small or when lit/FileCheck coverage is strong.

The hardware fixture set should be targeted to the active change. Running the
entire stress matrix is not required for every patch, but at least the relevant
existing or newly added hardware fixtures must run unless the patch is strictly
non-executable metadata, diagnostics, documentation, or verifier-only work. If
hardware is skipped for one of those reasons, the final report must state that
no executable lowering behavior changed.

Required fixture properties:

```text
- device-output comparison against CPU oracle
- sentinel preservation checks for stores
- zero-fill checks for loads
- row-tail cases
- column-tail cases once supported
- rectangular M/N tails
- K-tail contraction input zero-fill
- transposed/affine layout cases once supported
- full-tile fast-path cases proving dense paths do not degrade to sparse fallback
- sparse fallback cases if sparse fallback is implemented
```

Hardware is the gold standard. Lit/FileCheck tests are necessary but not
sufficient for any patch that affects executable predicated transfer/compute
behavior or changes which executable path a predicate uses.

---

## 9. Anti-shortcut rules

Implementations must not:

```text
- special-case fixture names, public names, expected JSON, or harness names
- fake VC4_TEST_RESULT
- alter expected JSON to match broken behavior
- precompute device results on the host in hardware fixtures
- lower VC4Tile directly to scheduled VC4
- lower predicates directly to SSAVC4 semantic predicate ops
- add producer lowering from Triton/IREE/JAX/PyTorch in this work
- implement executable sub-32 precision
- silently widen or pack unsupported precision paths
- add comments or string literals solely to satisfy verifier scans
- use physical QPU_NUMBER as logical predicate/program identity
```

---

## 10. Implementation order this document enables

After this document is checked in, the remaining action items should proceed in this order:

```text
2. Predicate Model
3. Predicate Algebra
4. Fragment Planner
5. Verifier Work
6. Diagnostics
7. VC4 Hardware Mapping
8. Memory Path Coverage
9. Layout Coverage
10. Compute Consumer Coverage
11. Tests
12. Cleanup / Architecture
13. Hardware Characterization
```

Hardware characterization should also run whenever an implementation decision depends on an unproven VC4 behavior.
Targeted hardware fixtures should run for each action item that changes
executable lowering, planner-to-lowering routing, hardware mapping, or fixture
expectations. The full matrix can be reserved for broader acceptance points.

---

## 11. Quick normative checklist

The most important locked semantics are:

```text
1. Predicates are logical-coordinate predicates, not raw vector<16xi1> masks.
2. Tile shapes stay static; predicates handle invalid/tail elements.
3. Loads zero-fill inactive coordinates and must not read inactive memory.
4. Stores preserve inactive destination coordinates and must not write inactive memory.
5. Shared-memory inactive behavior is explicit: zero_fill or preserve.
6. Dense rectangular/tail predicates are first-class.
7. Sparse masks are explicit fallback or deterministic rejection.
8. Predicates normalize into fragment plans before VC4Tile core verification.
9. VC4 supports structured 16-lane/VPM/VDR/VDW fragments, not arbitrary dynamic 2D mask tensors.
10. K-tail contraction inputs zero-fill; M/N-tail outputs preserve destination.
11. Predicate truth is layout-independent; layouts only control address mapping and fragment splitting.
12. M5/M6 v1 remains 32-bit-only for executable predicated transfer/compute.
```
