PHASE11_CONTROLLED_TRITON_STRIDED_MEMORY_FIXTURES=YES
PHASE11_VALUE_STRIDED_RANKED_MEMORY_CONTRACT=LOCKED
VALUE_RANK2_ROW_SLICE_TO_VC4KERNEL_STATIC=PASS
VALUE_MEMREF_DIM_METADATA_LOWERING=PASS
VALUE_STRIDED_RANKED_ADDRESS_PLANNER=YES
VALUE_STRIDED_RANKED_MEMORY_STATIC=PASS
VALUE_STRIDED_RANKED_MEMORY_HARDWARE_ISOLATION=PASS
VALUE_RANK1_FLATTENED_STRIDE_HARDWARE=PASS
VALUE_RANK2_IDENTITY_ROW_SLICE_HARDWARE=PASS
VALUE_RANK2_STRIDED_ROW_SLICE_HARDWARE=PASS
VALUE_MEMREF_DIM_METADATA_HARDWARE=PASS
REAL_TRITON_STRIDED_MEMORY_SOURCES=YES
REAL_TTIR_STRIDED_MEMORY_SNAPSHOTS=YES
ACCEPTED_FIXTURES_EXCLUDE_UNRELATED_STAGED_FEATURES=YES
RANK2_ROW_SLICE_IDENTITY_SURFACE=ACCEPTED
RANK2_ROW_SLICE_STRIDED_OUTER_DYNAMIC_SURFACE=ACCEPTED
MEMREF_DIM_METADATA_TO_SCALAR_ARG_CONTRACT=LOCKED
HIDDEN_MEMREF_DESCRIPTOR_ALLOWED=NO
GATHER_LANE_STRIDE_STAGED=YES
HIDDEN_MEMREF_DESCRIPTOR_REJECTED=YES
LANE_VARYING_STRIDE_GATHER_FIXTURE_STAGED=YES
COLUMN_SLICE_FIXTURE_STAGED=YES
READY_FOR_PHASE11_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_PHASE11_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_PHASE11_4_VALUE_RANKED_STRIDED_STATIC=YES
READY_FOR_PHASE11_3_VALUE_SURFACE_CONTRACT=YES
READY_FOR_TRITON=NO

# VC4 Vector/Triton Phase 11 Strided/Ranked Memory Fixtures

Phase 11.2 adds controlled real Triton sources and real emitted TTIR snapshots
for the strided/ranked memory skeleton feature band. These fixtures define the
Phase 11 TTIR acceptance contract; the earlier Phase 11.1 probes remain
inventory only.

Source-controlled fixture directory:

```text
examples/triton/phase11_strided_ranked_memory/
```

Accepted snapshots:

- `ttir_strided_row_copy_b16`: scalar `row * stride` plus contiguous
  `pid_col * 16 + tl.arange(0, 16)` row-slice copy.
- `ttir_strided_row_add_b16`: multi-buffer row-slice add with the same
  scalar-row-base plus contiguous-lane pointer shape for A, B, and C.
- `ttir_strided_row_inout_tail_b16`: input/output row-slice load/update/store
  with canonical Phase 10 tail masks.
- `mixed_ttir_strided_memory_axes_mask_cf_b16`: combines axis0 column blocks,
  axis1 rows, canonical tail masks, `tl.where`, and scalar control flow.

Staged snapshots:

- `ttir_lane_varying_stride_gather_reject_b16`: uses `lanes * stride`, which is
  lane-varying inner stride/gather and remains staged.
- `ttir_column_slice_reject_b16`: maps contiguous lanes to row coordinates and
  multiplies those lanes by `lda`/`ldo`, so it is a non-contiguous column slice
  and remains staged.

The accepted snapshots intentionally exclude unrelated staged features:
`sitofp`, `fptosi`, dot, reduce, block pointers, backend GPU dialects, subword
storage, and SFU/math expansion are absent. Triton frontend integer overflow
proof scaffolding such as `arith.extsi`, `arith.cmpi`, and `arith.andi` remains
normal emitted TTIR provenance from prior phases and is not a new source
feature.

Current value-import smoke records `EXPECTED_PENDING_IMPORTER_REPAIR`: the
existing importer still rejects multiple distinct pointer offset expressions
before Phase 11 implementation work. That is the expected boundary for this
fixture-design phase.

Phase 11.3 locks the value-surface counterpart: rank-1 flattened scalar
strided address skeletons, rank-2 identity row-slice transfers, rank-2
`strided<[?, 1], offset: 0>` row-slice transfers with explicit shape/stride
scalar metadata, and `memref.dim` metadata lowering to scalar shape args.
Lane-varying stride/gather and column-slice forms remain staged.

Phase 11.4 statically lowers the value subset through VC4Kernel, SSAVC4,
scheduled VC4, and bundle emission for non-hardware candidates. The value
address planner computes rank-1 flattened and rank-2 row-slice scalar element
base indices centrally, reuses the Phase 10 full/empty/tail mask classifier,
and rejects gather-like lane-varying stride, non-unit inner stride, column
slices, hidden descriptor metadata, and rank-2 tile/vector forms. Hardware
proof is intentionally deferred to Phase 11.5.

Phase 11.5 proves the value subset on VC4 hardware with targeted isolation
fixtures for rank-1 flattened row stride, rank-2 identity row slices, rank-2
strided row slices, `memref.dim` metadata row bounds, and repeated empty/non-
empty ranked launches. The fixtures use active_qpus=12, strict CPU oracles,
row-padding sentinels, output hashes, and checked `saw_*` claims.

`READY_FOR_TRITON=NO` remains locked.
