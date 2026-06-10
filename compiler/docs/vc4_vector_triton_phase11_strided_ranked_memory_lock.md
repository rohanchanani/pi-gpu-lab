# VC4 Vector/Triton Phase 11 Strided/Ranked Memory Final Lock

Phase 11 locks the value and TTIR strided/ranked memory skeleton subset. The
accepted executable surface covers rank-1 flattened scalar strided addresses,
rank-2 row-slice identity and `strided<[?, 1], offset: 0>` value memory,
metadata-only `memref.dim` lowering to scalar extent args, and real TTIR
row-strided pointer expressions of the form scalar row/column base plus a
contiguous `tt.make_range(0, 16)`.

The final TTIR mixed fixture is
`mixed_ttir_strided_memory_axes_mask_cf_b16_vc4triton`. It uses a
source-controlled real Triton source and real emitted TTIR snapshot, enters
through the C++ TTIR importer, lowers through the standard value layer and the
central value address planner, and runs on hardware with `active_qpus=12`,
strict CPU oracle checks, row-padding sentinels, and nonzero output hash
checking.

Gather/lane-varying stride, column/vertical slices, rank-2 vector/tile
transfers, hidden memref descriptors, block pointers, reductions, dot, GEMV,
GEMM, numeric casts, subword/f16 expansion, and SFU/math expansion remain
staged or rejected. `READY_FOR_TRITON=NO` remains locked.

PHASE11_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_STRIDED_RANKED_MEMORY_SKELETONS
VALUE_STRIDED_RANKED_MEMORY_CONTRACT=LOCKED
VALUE_STRIDED_RANKED_ADDRESS_PLANNER=YES
VALUE_RANK2_ROW_SLICE_TO_VC4KERNEL_STATIC=PASS
VALUE_MEMREF_DIM_METADATA_LOWERING=PASS
VALUE_STRIDED_RANKED_MEMORY_HARDWARE_ISOLATION=PASS
VALUE_STRIDED_RANKED_MEMORY_MIXED_ACCEPTANCE=PASS
REAL_TRITON_STRIDED_MEMORY_SOURCES=YES
REAL_TTIR_STRIDED_MEMORY_SNAPSHOTS=YES
TTIR_STRIDED_MEMORY_IMPORTER_STATIC=PASS
TTIR_STRIDED_MEMORY_HARDWARE_ISOLATION=PASS
TTIR_STRIDED_MEMORY_MIXED_ACCEPTANCE=PASS
GATHER_LANE_STRIDE_STAGED=YES
COLUMN_SLICE_STAGED=YES
HIDDEN_MEMREF_DESCRIPTOR_REJECTED=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_STRIDED_MEMORY_WORKAROUNDS=YES
READY_FOR_PHASE12_REDUCTIONS=YES
READY_FOR_TRITON=NO
