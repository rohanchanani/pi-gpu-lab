PHASE10_CONTROLLED_TRITON_MASK_MEMORY_FIXTURES=YES
REAL_TRITON_MASK_MEMORY_SOURCES=YES
REAL_TTIR_MASK_MEMORY_SNAPSHOTS=YES
ACCEPTED_FIXTURES_EXCLUDE_UNRELATED_STAGED_FEATURES=YES
PHASE10_VALUE_MASK_CLASSIFIER_CONTRACT=LOCKED
PHASE10_VALUE_MEMORY_LEGALITY_CONTRACT=LOCKED
SPARSE_MEMORY_MASKS_STAGED=YES
NONZERO_LOAD_OTHER_STAGED=YES
RANK2_STRIDED_MEMORY_STAGED=YES
SPARSE_STORE_MASK_FIXTURE_STAGED=YES
NONZERO_OTHER_LOAD_FIXTURE_STAGED=YES
READY_FOR_PHASE10_4_VALUE_MASK_MEMORY_CLASSIFIER_STATIC=YES
READY_FOR_PHASE10_3_VALUE_SURFACE_CONTRACT=YES
READY_FOR_TRITON=NO

# Phase 10 Mask/Memory Controlled Fixtures

Phase 10.2 locks the controlled real Triton source fixtures and source-controlled
TTIR snapshots for the mask classifier and memory-legality phase.

The fixtures are under:

```text
examples/triton/phase10_mask_memory/
```

They were emitted from real Triton v3.7.0 through the existing zero-setup
generator wrapper. No compiler semantic changes are part of this phase.

## Accepted Controlled Fixtures

Accepted lowerable snapshots:

- `ttir_mask_tail_load_store_b16.ttir.mlir`
- `ttir_mask_full_no_mask_b16.ttir.mlir`
- `ttir_mask_compute_select_tail_b16.ttir.mlir`
- `mixed_ttir_mask_memory_cf_axes_b16.ttir.mlir`

Accepted memory forms are flattened rank-1 identity `f32` transfers with
`BLOCK_SIZE=16`. Accepted memory masks are either absent for a known full-block
fixture shape or canonical `offsets < n` tail masks. Accepted compute masks may
feed `tl.where` / `arith.select`, but they do not feed `tt.load` or `tt.store`
mask operands.

The mixed fixture combines Phase 10 accepted memory masks with already locked
Phase 8.5 scalar control flow and Phase 9 multi-axis launch flattening:

```text
block_id = pid1 * tl.num_programs(0) + pid0
idx = block_id * 16 + lane
```

## Staged Controlled Fixtures

Staged snapshots:

- `ttir_mask_sparse_store_reject_b16.ttir.mlir`
- `ttir_mask_nonzero_other_reject_b16.ttir.mlir`

The sparse store fixture uses a data-dependent vector compare as the `tt.store`
mask. It does not rely on boolean `arith.andi` as the only blocker.

The nonzero-other fixture uses a canonical tail mask but sets `tl.load(...,
other=1.0)`. Phase 10 accepted load policy remains inactive zero / `other=0`.

## Excluded Unrelated Features

Accepted fixtures exclude rank-2/strided/block-pointer memory, gather, scatter,
dot, reduce, subword, SFU/math, `arith.sitofp`, and `arith.fptosi`.

Real Triton emits overflow guard compares and `arith.andi` operations around
offset construction. Those are not semantic transfer masks; Phase 10 mask
classification should inspect the SSA values used by `tt.load` and `tt.store`
mask operands.

## Current Import Smoke

Current value-import smoke is intentionally recorded as
`EXPECTED_PENDING_IMPORTER_REPAIR`. Tail, compute-select, and mixed accepted
snapshots already import. The accepted no-mask full-transfer snapshot currently
stages at the importer because no-mask `tt.store` support is not implemented
yet.

The staged snapshots produce precise staged diagnostics for sparse memory masks
and nonzero load `other`.

## Value Contract Delta

Phase 10.3 locks the corresponding value-surface contract before executable
classifier implementation:

- `FULL`, `EMPTY`, and `TAIL_0_TO_16` memory masks are the accepted transfer
  mask classes.
- `COMPUTE_MASK` is accepted only when the mask feeds compute operations such
  as `arith.select`, not transfer memory predicates.
- `SPARSE_OR_UNKNOWN_MEMORY_MASK` transfer masks are staged for loads and
  deterministic staged/reject for stores.
- `RECT` is staged until rank-2/tile memory phases.

Accepted value memory legality is rank-1 identity `#vc4value.global` memory
with `i32`/`f32`, `vector<16xi32>`/`vector<16xf32>` transfers, scalar base
indices, zero load `other`, inactive-zero loads, and inactive-preserve stores.

Nonzero load `other`, non-identity maps, rank-2/strided memory, gather/scatter,
block pointers, subword/f16 memory, boundary-check/padding-option producer
semantics, and vector rank greater than 1 remain staged.

`READY_FOR_TRITON=NO` remains locked.
