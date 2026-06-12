PHASE13_CONTROLLED_TRITON_GEMV_ROWWISE_DOT_FIXTURES=YES
REAL_TRITON_GEMV_ROWWISE_DOT_SOURCES=YES
REAL_TTIR_GEMV_ROWWISE_DOT_SNAPSHOTS=YES
PHASE13_VALUE_GEMV_ROWWISE_DOT_CONTRACT=LOCKED
VALUE_GEMV_ROWWISE_DOT_STATIC=PASS
VALUE_GEMV_F32_ROW_DOT_STATIC=PASS
VALUE_GEMV_I32_ROW_DOT_STATUS=STAGED_BY_I32_POLICY
VALUE_GEMV_PARTIAL_KBLOCK_STATIC=PASS
VALUE_GEMV_ROWWISE_DOT_HARDWARE_ISOLATION=PASS
VALUE_GEMV_F32_ROW_DOT_HARDWARE=PASS
VALUE_GEMV_I32_ROW_DOT_HARDWARE=NOT_REQUIRED_STAGED_BY_I32_POLICY
VALUE_GEMV_PARTIAL_KBLOCK_HARDWARE=PASS
VALUE_GEMV_EMPTY_REPEAT_HARDWARE=PASS
VALUE_GEMV_ROWWISE_DOT_F32_SURFACE=ACCEPTED
VALUE_GEMV_ROWWISE_DOT_I32_SURFACE=ACCEPTED
ACCEPTED_FIXTURES_EXCLUDE_UNRELATED_STAGED_FEATURES=YES
F32_DOT_FINITE_TREE_POLICY=YES
TL_DOT_TT_DOT_STAGED=YES
TL_DOT_TT_DOT_FIXTURE_STAGED=YES
MULTIBLOCK_K_ACCUMULATION_FIXTURE_STAGED=YES
MULTIBLOCK_K_ACCUMULATION_STAGED=YES
VECTOR_CONTRACT_STAGED=YES
READY_FOR_PHASE13_3_VALUE_SURFACE_CONTRACT=YES
READY_FOR_PHASE13_4_VALUE_GEMV_STATIC=YES
READY_FOR_PHASE13_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_PHASE13_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_TRITON=NO

# VC4 Vector/Triton Phase 13 GEMV Row-Wise Dot Fixtures

Phase 13.2 adds controlled real Triton sources and source-controlled emitted
TTIR snapshots for GEMV-v0 / row-wise dot. These fixtures define the TTIR
acceptance contract for the next value-surface contract package. They do not
modify compiler lowering and do not claim hardware proof.

## Accepted Scope

Accepted Phase 13.2 fixtures use:

- `tl.sum(a * x, axis=0)`, not `tl.dot`;
- `BLOCK_SIZE = 16`;
- row-strided A memory from Phase 11;
- rank-1 contiguous X memory;
- scalar Y or partial-output stores;
- Phase 10 canonical tail masks and zero `other`;
- Phase 12 add reductions;
- finite-input finite-tree f32 policy, not exact IEEE f32 reduction.

Accepted snapshots are:

- `ttir_gemv_row_dot_f32_b16.ttir.mlir`
- `ttir_gemv_row_dot_tail_f32_b16.ttir.mlir`
- `ttir_gemv_partial_kblock_f32_b16.ttir.mlir`
- `mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16.ttir.mlir`
- `ttir_gemv_row_dot_i32_b16.ttir.mlir`

The mixed fixture combines Phase 8.5 scalar control flow, Phase 9 axes, Phase
10 masks, Phase 11 row-strided memory, Phase 12 reductions, and the Phase 13
row-dot composite.

## Value Contract Handoff

PHASE13_VALUE_GEMV_ROWWISE_DOT_CONTRACT=LOCKED
VALUE_GEMV_ROWWISE_DOT_STATIC=PASS
VALUE_GEMV_F32_ROW_DOT_STATIC=PASS
VALUE_GEMV_I32_ROW_DOT_STATUS=STAGED_BY_I32_POLICY
VALUE_GEMV_PARTIAL_KBLOCK_STATIC=PASS
VALUE_GEMV_ROWWISE_DOT_HARDWARE_ISOLATION=PASS
VALUE_GEMV_F32_ROW_DOT_HARDWARE=PASS
VALUE_GEMV_I32_ROW_DOT_HARDWARE=NOT_REQUIRED_STAGED_BY_I32_POLICY
VALUE_GEMV_PARTIAL_KBLOCK_HARDWARE=PASS
VALUE_GEMV_EMPTY_REPEAT_HARDWARE=PASS
VALUE_GEMV_ROWWISE_DOT_F32_SURFACE=ACCEPTED
VALUE_GEMV_ROWWISE_DOT_I32_SURFACE=ACCEPTED
F32_DOT_FINITE_TREE_POLICY=YES
TL_DOT_TT_DOT_STAGED=YES
VECTOR_CONTRACT_STAGED=YES
MULTIBLOCK_K_ACCUMULATION_STAGED=YES
READY_FOR_PHASE13_4_VALUE_GEMV_STATIC=YES
READY_FOR_PHASE13_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_PHASE13_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_TRITON=NO

Phase 13.3 locks the value form that corresponds to these controlled TTIR
fixtures: elementwise multiply over `vector<16>` followed by add reduction and
scalar store. F32 dots use finite-input finite-tree policy. I32 row dots remain
surface-admissible, but executable lowering is staged by the current i32
multiply policy after Phase 13.5 hardware isolation showed that `mul24_safe` is
not an exact signed i32 dot policy. K is limited to one `vector<16>` block for
full row-dot outputs; partial K-block dots store one independent scalar partial
and do not accumulate across K blocks.

Phase 13.4 statically proves the accepted value composite. The path remains
standard value IR to VC4Kernel through existing fragment multiply, add
reduction, and scalar-store lowerers. `tl.dot`, `tt.dot`, `vector.contract`, and
multi-block K accumulation remain staged. No hardware proof is claimed by this
static package.

Phase 13.5 hardware isolation proves the f32 row-wise dot, partial K-block dot,
and empty/repeat launch value fixtures on hardware with `active_qpus=12`,
sentinels, and CPU oracles. I32 row-dot hardware is not required because the
current executable i32 multiply policy is staged for dot composites after the
strict isolation fixture exposed non-exact signed i32 products.

## Staged Scope

`ttir_tl_dot_reject_b16.ttir.mlir` is staged because real Triton emits
first-class `tt.dot` over rank-2 tensor operands. Phase 13 GEMV-v0 does not
accept `tl.dot`, `tt.dot`, `vector.contract`, or rank-2 tile values.

`ttir_gemv_loop_kblocks_reject_b16.ttir.mlir` is staged because it performs
multi-block K accumulation through a loop-carried scalar f32 accumulator. Phase
13 GEMV-v0 accepts one `vector<16>` K block or one independently stored partial
K-block result, not cross-block accumulation.

`vector.contract` remains staged. It is a value-layer operation, not a real
Triton TTIR snapshot form in this package.

## Files

Sources and snapshots live under:

```text
examples/triton/phase13_gemv_rowwise_dot/
```

The manifest records generator commands, classifications, feature claims, and
the f32 finite-tree policy caveat:

```text
examples/triton/phase13_gemv_rowwise_dot/manifest.json
```

The controlled inventory for this package is:

```text
.vc4_auto/vector_triton_phase13_2_controlled_fixtures/CONTROLLED_TTIR_GEMV_INVENTORY.md
```

## Handoff

Phase 13.6 should add value mixed acceptance fixtures for the accepted row-wise
dot forms. `READY_FOR_TRITON=NO` remains true.
