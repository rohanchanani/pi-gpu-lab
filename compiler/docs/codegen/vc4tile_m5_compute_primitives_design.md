# VC4Tile M5 Compute-Primitives Decision: Tile Contract, Dot, Reductions, and Companion Tile Movement Ops

**Date:** 2026-05-27  
**Status:** locked design decision for the first M5 big-ticket item  
**Scope:** VC4Tile ergonomic surface operations for tile-level computation and their relationship to Triton and IREE/Linalg producer lowering  
**Non-scope:** mixed precision / packing policy, the full copy-planner algorithm, and actual Triton/IREE adapter implementation are separate design items

---

## 1. Original question

We needed to decide whether VC4Tile should grow high-level computation primitives such as `tile_dot`, `tile_matmul`, `tile_contract`, `tile_reduce`, and `block_reduce`.

The central concern was not whether these operations are useful in the abstract. The question was whether they are useful **as lowering targets for future upstream producers**:

```text
Triton MLIR / TTIR
StableHLO -> IREE -> Linalg / late executable IR
    ↓
VC4Tile ergonomic surface
    ↓ canonicalize-vc4tile-surface
VC4Tile core
    ↓ convert-vc4tile-to-ssavc4
SSAVC4
    ↓ convert-ssavc4-to-vc4
scheduled VC4
    ↓ vc4-codegen
QASM / runtime / hardware
```

The competing possibilities were:

1. Add `tile_dot` / `tile_contract` / reductions to the ergonomic VC4Tile layer because Triton and IREE/Linalg naturally preserve contraction/reduction semantics that need a target-specific tile implementation point.
2. Do **not** add them because future Triton/IREE lowering would bypass them and lower directly into core VC4Tile primitives anyway.
3. Add them to a different level of abstraction, such as SSAVC4 or a future producer-specific dialect, instead of VC4Tile.

The decision below resolves that question.

---

## 2. What the decision must satisfy

The design must satisfy five constraints simultaneously.

### 2.1 It must be a natural target for Triton

Triton IR is already tile/block oriented. It has program IDs, block tensors, pointer/tensor loads and stores, reductions, and a first-class `tt.dot` operation. A useful VC4Tile surface should preserve that semantic structure long enough for VC4-specific tile planning, instead of immediately exploding a contraction into ad hoc scalar/vector loops in every producer adapter.

### 2.2 It must be a natural target for IREE/Linalg

The JAX/PyTorch path through IREE will not literally emit VC4Tile operations. It will expose high-level StableHLO operations and then internal IREE/Linalg/vector/tiled executable forms. The VC4 backend needs a target-specific convergence point for tiled contractions, reductions, promotion/copy decisions, and shared-memory use. VC4Tile surface ops should provide that convergence point.

### 2.3 It must lower cleanly to VC4Tile core

VC4 has no tensor cores and no native matrix-multiply instruction. Therefore `tile_contract` / `tile_dot` must **not** be treated as opaque core operations that lower directly to SSAVC4 as if the hardware had an MMA primitive. They must canonicalize into explicit VC4Tile core operations: arithmetic, `scf`/`cf` control flow as legal after the SCF/Core-CFG boundary work, masks, global/shared movement, reductions, barriers, and existing core memory primitives.

### 2.4 It must respect the existing lower-half boundary

SSAVC4 is a pre-register-allocation, pre-scheduling machine-SSA IR, and scheduled `vc4` is the post-register-allocation/post-scheduling QASM-near sink. The lower half should not learn high-level algorithmic operations such as `ssavc4.matmul` or `ssavc4.conv`. VC4Tile is the right layer for tile-level semantic operations; SSAVC4 remains a lower target-machine layer.

### 2.5 It must be extensible without freezing future work

M5 should not attempt to implement every possible CuTe/ThunderKittens feature in one shot. But the interfaces added now must not block later additions such as richer layouts, role annotations, copy-pipeline stages, mixed precision, packing, or more aggressive contraction implementations.

---

## 3. Evidence used in the decision

### 3.1 Triton evidence

The official Triton MLIR operation documentation contains first-class operations that directly correspond to the VC4Tile surface primitives we are considering:

- `tt.dot` is a first-class Triton dot operation with syntax taking `a`, `b`, and `c` and producing `d`.
- `tt.reduce` performs a generic reduction over an axis and has a reduction body terminator.
- `tt.load` loads from a pointer or tensor of pointers and supports masks/alternate values.
- `tt.store` stores through a pointer or tensor of pointers and supports a mask.

This means Triton lowering can naturally map:

```text
tt.load     -> vc4tile.tile_load or lower-level masked load if already scalarized
tt.dot      -> vc4tile.tile_contract / vc4tile.tile_dot
tt.reduce   -> vc4tile.tile_reduce / vc4tile.block_reduce
tt.store    -> vc4tile.tile_store or lower-level masked store
```

Triton also documents a `TensorDescType` for tiled tensor memory access, with shape, element type, and optional shared layout metadata. That is strong evidence that the VC4Tile surface should include layout-aware tile movement concepts, not just anonymous vector loads and stores.

### 3.2 StableHLO and IREE/Linalg evidence

StableHLO is explicitly a portability layer between ML frameworks and compilers, including frameworks such as JAX and compilers such as IREE. Its `dot_general` semantics describe general contraction through batching and contracting dimensions.

MLIR Linalg exposes contraction operations that are a closer match to backend tiling decisions:

- `linalg.matmul` is a named 2D matrix multiplication operation.
- `linalg.contract` expresses generalized contraction with explicit affine maps and reduction dimensions.
- `linalg.mmt4d` represents matrix multiplication over 4D tiled operands, with inner tile dimensions.

IREE’s GPU codegen pass list is also informative. It includes tiling, distribution, shared-memory copy distribution, matmul operand promotion, packing partial reductions, packing matmul-like operations to intrinsics, vector allocation, shared-memory reuse, and reduction tiling.

This does **not** mean VC4Tile should copy IREE’s GPU pipeline wholesale. It means future IREE lowering will naturally contain exactly the semantic units that VC4Tile surface ops should receive: tiled contractions, tiled reductions, shared/copy promotion decisions, and tile layouts.

### 3.3 VC4 hardware and project-boundary evidence

The local VC4 mapping guide models VC4 as one tiny CUDA-like SM with 12 physical warp slots, 16 SIMD lanes per warp, and one global 4 KiB user-visible VPM shared-memory window. That model supports tile-level programming and cooperative shared-memory behavior, but it does not imply a native matrix-multiply unit.

The SSAVC4 design boundary is also already established: SSAVC4 is the pre-register-allocation/pre-scheduling machine SSA layer; scheduled VC4 is the artifact sink. The lower half must remain target-machine IR and scheduled QPU IR, not a high-level algorithmic layer.

Therefore, computation primitives belong in the **VC4Tile ergonomic surface**, where they can be target-aware but still semantic enough to interface with Triton and IREE/Linalg.

---

## 4. The decision

### 4.1 Main decision

Add tile-level computation primitives to the ergonomic VC4Tile layer.

Specifically:

```text
M5 first half:
  tile fragments / tile descriptors / layout vocabulary
  tile_load
  tile_store
  copy_tile
  tile_reduce
  block_reduce

M5 second half:
  tile_contract
  tile_dot
  tile_matmul as optional sugar over tile_contract, not as a separate core concept
```

All of these are **surface operations** unless explicitly stated otherwise. They must canonicalize away before `--convert-vc4tile-to-ssavc4`.

The post-canonicalization legal pipeline is:

```text
producer-ish ergonomic VC4Tile surface
    ↓ --canonicalize-vc4tile-surface
VC4Tile core only
    ↓ --legalize-vc4tile-core-cfg if SCF was used
VC4Tile core CFG verified by --verify-vc4tile-core
    ↓ --convert-vc4tile-to-ssavc4
SSAVC4
```

No `tile_contract`, `tile_dot`, `tile_matmul`, `tile_load`, `tile_store`, or `copy_tile` surface op may remain after canonicalization unless the operation has explicitly been promoted into the core op set by a later, separately verified milestone.

### 4.2 Negative decision: do not add whole-tensor matmul to core

Do **not** add a general whole-problem `vc4tile.matmul` core op.

`linalg.matmul`, StableHLO `dot_general`, and Triton `tt.dot` operate at different abstraction levels. VC4Tile should receive a **tile-sized contraction fragment** after producer-side or backend-side tiling decisions, not a whole tensor matmul requiring global scheduling, decomposition, packing, and block mapping inside one monolithic op.

If a spelling named `vc4tile.tile_matmul` is added, it must be syntactic sugar for a constrained two-input contraction and canonicalize to `vc4tile.tile_contract` early.

### 4.3 Negative decision: do not put these in SSAVC4

Do **not** add `ssavc4.matmul`, `ssavc4.tile_dot`, `ssavc4.conv`, or similar high-level algorithmic operations.

SSAVC4 should continue to represent lower machine-SSA semantics: ALU, flags, rotates, VPM/TMU/VDW, barriers, branches, metadata, and effectful hardware operations. High-level contraction planning belongs above SSAVC4.

---

## 5. Why the operations should not be bypassed

A producer adapter *could* lower Triton `tt.dot` or IREE/Linalg contraction directly into core VC4Tile loops and loads. That is possible, but it is the wrong default architecture.

Direct-to-core lowering would force every producer adapter to duplicate VC4-specific decisions:

- how to choose tile shape within 16-lane SIMD constraints,
- how to map K reduction to lanes, loops, or block cooperation,
- whether to use VPM as shared memory,
- when to insert barriers,
- how to choose copy routes,
- how to split reductions across lanes or warps,
- what layout vocabulary applies,
- how to handle masks/tails,
- how to validate real hardware behavior.

A `tile_contract` / `tile_reduce` surface preserves the semantic intent and lets one VC4-owned canonicalization path implement those choices. That gives us a single place to tune the VC4 strategy while still accepting both Triton and IREE/Linalg inputs.

The adapter is allowed to bypass these ops only when the upstream operation is already low-level enough that no tile-level semantic decision remains. Example: a Triton elementwise kernel with `tt.load`, arithmetic, and `tt.store` may lower directly into existing core masked load/store/arithmetic, with no `tile_contract` involved.

---

## 6. Operation families and contracts

### 6.1 Tile/value model

M5 should introduce a surface notion of tile fragment. The exact syntax can be refined in the milestone package, but the semantic model should be:

```text
!vc4tile.tile<
  shape = [...],
  elem_type = ...,
  role = input | accumulator | output | scratch,
  layout = ...,
  memory = global | shared_vpm | register,
  scope = lane | warp | block
>
```

This type or descriptor is a surface carrier. It should be used to give the canonicalizer enough static information to generate core operations. It should not leak into SSAVC4. It should also not require the final register layout to be known at the surface layer.

Minimum M5 first-half requirements:

- support static rank/shape metadata for tile fragments;
- support element types currently executable in the core pipeline, primarily `i32`, `u32`-mode integer data, and `f32` where the current lower half supports it;
- support layout metadata sufficient for row-major, column-major, transposed, contiguous, strided, and VPM-compatible tile views;
- support role annotations as semantic metadata, even if only used by diagnostics/canonicalization at first;
- reject unsupported dynamic tile shapes unless a specific feature slice proves them.

### 6.2 `vc4tile.tile_load`

Purpose: move a logical tile fragment from global memory or shared VPM into a tile value.

Conceptual form:

```mlir
%tile = vc4tile.tile_load %base, %indices, %mask
  {layout = ..., elem_type = ..., shape = ..., memory_space = ..., cache/policy attrs...}
  : ... -> !vc4tile.tile<...>
```

Upstream mapping:

- Triton `tt.load` with tensor-of-pointers or descriptor-based tiled memory access can lower to `tile_load` when the load semantically produces a tile fragment.
- IREE/Linalg promotion, pack, or tiled input reads can lower to `tile_load` when loading a contraction/reduction tile.

Downward canonicalization:

- independent-vector loads may become core masked global loads plus arithmetic/masks;
- cooperative/shared loads may become a planned sequence involving global memory, VPM/shared tile storage, barriers, and core shared loads;
- unsupported layouts must fail with deterministic diagnostics, not silently degrade.

### 6.3 `vc4tile.tile_store`

Purpose: store a tile fragment to global memory or shared VPM.

Conceptual form:

```mlir
vc4tile.tile_store %tile, %base, %indices, %mask
  {layout = ..., memory_space = ..., write_policy = ...}
  : !vc4tile.tile<...>, ...
```

Upstream mapping:

- Triton `tt.store` maps naturally to `tile_store` when the stored value is tile-shaped.
- IREE/Linalg destination-style output tiles or epilogue stores can map to `tile_store`.

Downward canonicalization:

- logical global stores may canonicalize into core masked global store operations;
- if the hardware path requires staging through VPM/VDW, the canonicalizer/copy planner must expose that path in core operations or metadata;
- masks/tails must be preserved through hardware validation.

Important subtlety: `tile_store` is a logical surface operation. It does **not** assert that the hardware has a single direct register-to-global instruction. The canonicalized implementation may involve register-to-VPM/shared movement and VPM/VDW-backed global writeout.

### 6.4 `vc4tile.copy_tile`

Purpose: express a planned tile movement independent of the computation op.

Conceptual form:

```mlir
%dst = vc4tile.copy_tile %src
  {src_layout = ..., dst_layout = ..., src_memory = ..., dst_memory = ..., role = ...}
  : !vc4tile.tile<...> -> !vc4tile.tile<...>
```

Primary use cases:

- global-to-shared tile promotion;
- shared-to-register tile load;
- register-to-shared tile spill/staging;
- shared/VPM-to-global tile writeback;
- layout transform or transpose before a contraction;
- double-buffering hooks in a later milestone.

Interplay with computation primitives:

```text
tile_load / copy_tile prepare operand fragments
    ↓
tile_contract or tile_reduce consumes prepared fragments
    ↓
copy_tile / tile_store moves accumulator/output fragments back out
```

The computation op should not secretly own all data movement. It may request operand roles and layout constraints, but explicit movement operations should make the pipeline visible and testable.

This design deliberately leaves the full copy planner as a separate big-ticket design item. However, the M5 compute-primitives interface must be copy-planner-ready: every tile movement op needs enough shape/layout/memory/role metadata for the planner to choose legal VC4 paths.

### 6.5 `vc4tile.tile_reduce`

Purpose: reduce within a tile fragment or along a tile dimension.

Conceptual form:

```mlir
%out = vc4tile.tile_reduce %tile
  {kind = add | max | min | and | or | custom_later,
   axis = ..., layout = ..., mask_policy = ...}
  : !vc4tile.tile<...> -> !vc4tile.tile<...> or scalar/vector
```

Upstream mapping:

- Triton `tt.reduce` maps naturally to `tile_reduce` when reduction stays within a block/tile.
- Linalg reductions and IREE partial-reduction tiling can map to `tile_reduce` when the tile scope is local.

Downward canonicalization:

- lane-local reductions can use existing core rotate/reduce mechanisms;
- wider reductions can use shared VPM plus barriers when cooperative block resources are required;
- unsupported reduce kinds or axes must produce deterministic diagnostics.

### 6.6 `vc4tile.block_reduce`

Purpose: reduce across a cooperative block, potentially across multiple warps/QPUs.

Conceptual form:

```mlir
%out = vc4tile.block_reduce %value_or_tile
  {kind = ..., scope = block, algorithm_hint = ...}
  : ... -> ...
```

This should be included in M5 first half because it builds on already-proven lower-half machinery: rotate/reduce, shared VPM, cooperative resources, and barriers.

Upstream mapping:

- Triton block reductions and reduction epilogues can target this.
- IREE tiled reductions and partial-reduction combination can target this after distribution.

Downward canonicalization:

- one-warp reductions become warp/local reductions;
- multi-warp reductions use shared VPM and block barriers;
- full-block residency constraints must be explicit in resource metadata.

### 6.7 `vc4tile.tile_contract`

Purpose: represent a tile-sized contraction update.

Conceptual form:

```mlir
%d = vc4tile.tile_contract %a, %b, %c
  {m = ..., n = ..., k = ...,
   lhs_layout = ..., rhs_layout = ..., acc_layout = ...,
   lhs_role = input, rhs_role = input, acc_role = accumulator,
   contract_dims = ..., batch_dims = [],
   precision_policy = ...}
  : !vc4tile.tile<...>, !vc4tile.tile<...>, !vc4tile.tile<...>
    -> !vc4tile.tile<...>
```

Required semantics:

```text
D[m, n] = C[m, n] + sum_k A[m, k] * B[k, n]
```

Generalized forms may support transpose/broadcast/batch through layout and indexing metadata, but M5 second half should begin with the smallest hardware-provable subset.

Upstream mapping:

- Triton `tt.dot` maps to `tile_contract` when the producer has a dot-shaped tile operation.
- StableHLO `dot_general`, after IREE/Linalg tiling, maps to `tile_contract` from `linalg.matmul`, `linalg.contract`, or `linalg.mmt4d`-like forms.

Downward canonicalization:

- no `tile_contract` may survive into core lowering;
- canonicalization expands the contraction into explicit loops, arithmetic, tile loads/copies, accumulator updates, reductions if needed, masks/tails, and stores;
- the first implementation should prefer correctness and hardware proof over performance.

### 6.8 `vc4tile.tile_dot`

Purpose: convenience spelling for the common dot/contract case.

Decision:

```text
vc4tile.tile_dot is allowed as a surface alias.
vc4tile.tile_dot canonicalizes to vc4tile.tile_contract.
vc4tile.tile_dot must not become a separate core operation.
```

The name `tile_dot` is useful for Triton alignment because Triton has `tt.dot`, but internally `tile_contract` should be the canonical semantic op.

### 6.9 `vc4tile.tile_matmul`

Purpose: optional readability alias for matrix multiplication-shaped contraction.

Decision:

```text
vc4tile.tile_matmul may exist as sugar if it improves tests/readability.
It canonicalizes to vc4tile.tile_contract.
It must not imply whole-tensor matmul ownership.
```

If added, its verifier must reject dynamic whole-tensor semantics and require tile-fragment operands/results.

---

## 7. M5 staging plan

### 7.1 M5 first half: movement, fragments, and reductions

M5 first half should implement the substrate needed by contractions before implementing contractions themselves.

Required surface features:

1. Tile fragment type/descriptor syntax.
2. Layout vocabulary sufficient for row/column/transposed/strided/VPM-compatible fragments.
3. `tile_load`.
4. `tile_store`.
5. `copy_tile`.
6. `tile_reduce`.
7. `block_reduce`.
8. Canonicalization pass that eliminates all first-half surface ops into VC4Tile core.
9. Core verification that rejects remaining surface ops before SSAVC4 lowering.
10. Hardware fixtures for each meaningful feature family.

Representative first-half fixtures:

```text
1. tile_load_store_identity_i32_vc4tile
2. tile_load_store_tail_mask_vc4tile
3. tile_copy_global_to_shared_to_global_vc4tile
4. tile_copy_transpose_shared_vc4tile
5. tile_reduce_sum_warp_vc4tile
6. tile_reduce_max_warp_vc4tile
7. block_reduce_sum_two_warps_vc4tile
8. block_reduce_sum_full_block_vc4tile
9. tile_store_epilogue_masked_vc4tile
10. tile_layout_invalid_stride_vc4tile
```

The exact fixture names can change, but the coverage requirement should not: at least one or two real hardware fixtures per significant operation family, with device-derived output comparisons.

### 7.2 M5 second half: contraction/dot/matmul sugar

M5 second half should implement contraction only after movement/layout/reduction primitives exist.

Required surface features:

1. `tile_contract` as the canonical contraction operation.
2. `tile_dot` as an alias/canonicalization source for Triton alignment.
3. Optional `tile_matmul` sugar, only if it canonicalizes immediately to `tile_contract`.
4. Canonicalization of contraction into first-half movement/reduction/core operations, and then into VC4Tile core.
5. Hardware fixtures proving contraction correctness on real VC4 hardware.
6. Diagnostics rejecting unsupported shapes/layouts/element types.

Representative second-half fixtures:

```text
1. tile_contract_1x16_dot_i32_vc4tile
2. tile_contract_4x4_f32_smoke_vc4tile
3. tile_contract_transposed_rhs_vc4tile
4. tile_contract_tail_k_vc4tile
5. tile_contract_shared_operand_promotion_vc4tile
6. tile_dot_alias_canonicalizes_vc4tile
7. tile_matmul_alias_canonicalizes_vc4tile
8. tile_contract_invalid_dynamic_shape_vc4tile
9. tile_contract_invalid_layout_vc4tile
10. attention_qk_tiny_tile_contract_vc4tile
```

Again, names may change, but hardware proof should be extensive and real.

---

## 8. Interface up: how producers should target these ops

### 8.1 Triton path

A future Triton adapter should preserve the following semantic mapping when possible:

```text
tt.get_program_id        -> vc4tile.program_id / block_id mapping as applicable
tt.arange / make_range   -> lane_range / tile index construction
tt.load                  -> tile_load or core masked load
tt.store                 -> tile_store or core masked store
tt.reduce                -> tile_reduce or block_reduce
tt.dot                   -> tile_dot -> tile_contract
tt.descriptor_*          -> tile descriptors / tile_load / copy_tile where applicable
```

The adapter should not expand `tt.dot` directly into low-level loops unless the shape is outside the accepted `tile_contract` subset or a specific optimization path is intentionally selected and separately tested.

### 8.2 IREE / JAX / PyTorch path

A future IREE adapter should lower from a late IREE/Linalg/vector/tiled executable form, not from raw high-level framework tensors directly into VC4Tile.

Expected mapping:

```text
StableHLO dot_general
  -> IREE/Linalg contraction form
  -> linalg.matmul / linalg.contract / linalg.mmt4d or tiled equivalent
  -> vc4tile.tile_contract

Linalg reductions / IREE tiled reductions
  -> vc4tile.tile_reduce or vc4tile.block_reduce

IREE matmul operand promotion / shared copy distribution
  -> vc4tile.tile_load / copy_tile / tile_store
```

The VC4 backend may also consume lower-level vector/tiled loops directly when no contraction semantic remains, but preserving contraction as `tile_contract` is preferred when the upstream IR still makes the contraction evident.

---

## 9. Interface down: canonicalization into VC4Tile core

The canonicalization pass must be the single enforcement point for this rule:

```text
No surface tile operation reaches --convert-vc4tile-to-ssavc4.
```

A good verification sequence is:

```text
vc4-opt input.mlir \
  --canonicalize-vc4tile-surface \
  --legalize-vc4tile-core-cfg \
  --verify-vc4tile-core \
  --convert-vc4tile-to-ssavc4
```

Canonicalization must produce only operations that the existing core/lower-half can handle or that are added as core operations in the same milestone with full tests.

Surface-to-core examples:

```text
tile_load(global, row_major, mask)
  -> address arithmetic + lane_range + tail_mask + masked_load_global

copy_tile(global -> shared_vpm)
  -> planned global load + shared/VPM store + barrier if cooperative

tile_reduce(add, lane)
  -> rotate/reduce core operations

block_reduce(add)
  -> warp reduce + shared VPM staging + barrier + final warp reduce

tile_contract
  -> loops over K + tile_load/copy_tile + per-lane multiply-add + reductions/stores
```

The first implementation may choose conservative expansion and full unrolling for static small shapes. Performance improvements can follow only after hardware correctness is established.

---

## 10. Layout and copy-planner requirements imposed by this decision

This document does not fully design the copy planner. However, computation primitives impose minimum requirements on it.

The copy planner must eventually be able to reason about:

- source memory: global, shared VPM, register tile;
- destination memory: global, shared VPM, register tile;
- shape and element type;
- layout: row-major, column-major, transposed, strided, VPM row/column forms;
- tile role: LHS, RHS, accumulator, epilogue output, scratch;
- masks and tails;
- cooperative scope and required barriers;
- whether a copy can use an existing core masked load/store or must stage through VPM/VDW;
- whether a copy is legal on VC4 hardware.

The computation primitives should expose enough metadata for the planner to make those choices. They should not hide the copy pipeline inside `tile_contract`.

Preferred pattern:

```mlir
%a_g = vc4tile.tile_load ... {role = lhs, memory = global, layout = ...}
%b_g = vc4tile.tile_load ... {role = rhs, memory = global, layout = ...}
%a_s = vc4tile.copy_tile %a_g {dst_memory = shared_vpm, dst_layout = ...}
%b_s = vc4tile.copy_tile %b_g {dst_memory = shared_vpm, dst_layout = ...}
%c   = vc4tile.tile_init ... {role = accumulator}
%d   = vc4tile.tile_contract %a_s, %b_s, %c { ... }
vc4tile.tile_store %d, ... {memory = global, layout = ...}
```

A simpler independent-vector kernel may omit explicit shared copies and lower directly through masked global loads/stores.

---

## 11. Precision policy for this decision

Mixed precision, packing, FP16/FP8/INT8, and VPM packing are important and should be designed deliberately. They are **not** resolved by this document.

However, the interfaces added here should be future-proof:

- tile fragment metadata should include `elem_type` and `accumulator_type` fields;
- `tile_contract` should include a `precision_policy` or equivalent attribute, even if only a conservative subset is initially accepted;
- diagnostics must reject unsupported precision/layout combinations instead of silently changing semantics;
- canonicalization must not assume all future contractions are `f32 x f32 -> f32`.

Initial M5 implementations should use element types already proven by the core/lower-half pipeline unless the mixed-precision design item has been completed and accepted.

---

## 12. Verification and anti-shortcut requirements

Every feature slice implementing these operations must include:

1. Dialect roundtrip tests.
2. Invalid diagnostic tests.
3. Canonicalization tests proving surface ops disappear.
4. Core verifier tests proving remaining core is legal.
5. Lowered SSAVC4 lit tests.
6. Scheduled VC4 / artifact tests when relevant.
7. Real hardware fixtures for executable features.
8. Device-derived output checking in candidate harnesses.
9. Negative scans preventing fake `VC4_TEST_RESULT` or fixture-name special casing.
10. Regression against M2/M3/M4 acceptance as appropriate.

Hardware fixtures are mandatory for all executable compute features. A green lit test is not enough.

A fixture should be considered insufficient if:

- it only checks that the compiler emitted some operation names;
- it compares against host-computed output without copying device output back;
- it hard-codes `status=PASS`;
- it only tests a trivial all-zero input that could pass through a broken computation path;
- it tests one exact shape while the verifier accepts broader shapes that are not semantically correct.

---

## 13. Acceptance criteria for this design item

The design item is accepted when the milestone package and implementation plan encode the following facts:

1. VC4Tile surface will include data movement, reduction, and contraction primitives.
2. `tile_load`, `tile_store`, `copy_tile`, `tile_reduce`, and `block_reduce` are first-half M5 features.
3. `tile_contract`, `tile_dot`, and optional `tile_matmul` sugar are second-half M5 features.
4. `tile_contract` is the canonical contraction semantic operation.
5. `tile_dot` and `tile_matmul` canonicalize to `tile_contract`; they are not separate core concepts.
6. No surface operation reaches SSAVC4 lowering.
7. Triton `tt.dot` / `tt.reduce` and IREE/Linalg contraction/reduction paths are explicit upstream motivations.
8. The downward lowering path is to VC4Tile core, not to SSAVC4 algorithmic ops.
9. Copy movement primitives are explicit companions to computation primitives.
10. Mixed precision remains a separate design item, but the interfaces are not hostile to future precision support.
11. Hardware fixtures are required and must validate real device output.

---

## 14. Summary decision statement

VC4Tile should grow toward CuTe/ThunderKittens-style tile ergonomics, but in a VC4-specific and compiler-verifiable way.

For the first big-ticket item, we will add a surface-level tile computation layer that future Triton and IREE/Linalg lowering can target. The key abstraction is `tile_contract`, with `tile_dot` as a Triton-aligned alias and optional `tile_matmul` as sugar. Reductions and tile movement operations are not optional extras; they are the substrate that makes contraction meaningful and implementable.

These operations are not hardware opcodes. They are semantic surface operations. Their job is to preserve upstream computation intent long enough for VC4-specific planning, then canonicalize into proven VC4Tile core operations that already know how to lower through SSAVC4 and scheduled VC4 to real hardware.

---

## References

- Triton MLIR `TritonOps` documentation: `tt.dot`, `tt.reduce`, `tt.load`, and `tt.store`. <https://triton-lang.org/main/dialects/TritonOps.html>
- Triton MLIR `tt` dialect documentation: `TensorDescType` for tiled tensor memory access. <https://triton-lang.org/main/dialects/TritonDialect.html>
- StableHLO specification: `dot_general` and StableHLO as a portability layer for ML frameworks and compilers. <https://openxla.org/stablehlo/spec>
- MLIR Linalg dialect documentation: `linalg.matmul`, `linalg.contract`, and `linalg.mmt4d`. <https://mlir.llvm.org/docs/Dialects/Linalg/>
- IREE Common/GPU codegen pass documentation: tiling, shared-memory copy distribution, matmul operand promotion, partial reductions, packing, and vector allocation. <https://iree.dev/reference/mlir-passes/CodegenCommonGPU/>
- IREE Global Optimization pass documentation: contraction demotion, linalg named-op generalization, matmul/dequantization fusion. <https://iree.dev/reference/mlir-passes/GlobalOptimization/>
- Local project context: `vc4_cuda_mapping_guide.md`, `vc4tile-ir-design-m4.md`, `ssavc4-ir-design-m3-post-cleanup.md`, and related M2/M3/M4 milestone documentation.
