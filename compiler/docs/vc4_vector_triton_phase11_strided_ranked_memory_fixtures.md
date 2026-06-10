PHASE11_CONTROLLED_TRITON_STRIDED_MEMORY_FIXTURES=YES
REAL_TRITON_STRIDED_MEMORY_SOURCES=YES
REAL_TTIR_STRIDED_MEMORY_SNAPSHOTS=YES
ACCEPTED_FIXTURES_EXCLUDE_UNRELATED_STAGED_FEATURES=YES
LANE_VARYING_STRIDE_GATHER_FIXTURE_STAGED=YES
COLUMN_SLICE_FIXTURE_STAGED=YES
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

`READY_FOR_TRITON=NO` remains locked.
