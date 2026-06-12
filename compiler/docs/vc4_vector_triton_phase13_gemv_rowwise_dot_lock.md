# VC4 Vector/Triton Phase 13 GEMV Row-Wise Dot Final Lock

PHASE13_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_GEMV_ROWWISE_DOT
VALUE_GEMV_ROWWISE_DOT_CONTRACT=LOCKED
VALUE_GEMV_ROWWISE_DOT_STATIC=PASS
VALUE_GEMV_F32_ROW_DOT_STATIC=PASS
VALUE_GEMV_I32_ROW_DOT_STATUS=STAGED_BY_I32_POLICY
VALUE_GEMV_ROWWISE_DOT_HARDWARE_ISOLATION=PASS
VALUE_GEMV_ROWWISE_DOT_MIXED_ACCEPTANCE=PASS
REAL_TRITON_GEMV_ROWWISE_DOT_SOURCES=YES
REAL_TTIR_GEMV_ROWWISE_DOT_SNAPSHOTS=YES
TTIR_GEMV_ROWWISE_DOT_IMPORTER_STATIC=PASS
TTIR_GEMV_ROWWISE_DOT_HARDWARE_ISOLATION=PASS
TTIR_GEMV_ROWWISE_DOT_MIXED_ACCEPTANCE=PASS
F32_DOT_FINITE_TREE_POLICY=YES
EXACT_F32_DOT_NOT_CLAIMED=YES
TL_DOT_TT_DOT_STAGED=YES
VECTOR_CONTRACT_STAGED=YES
MULTIBLOCK_K_ACCUMULATION_STAGED=YES
FULL_GEMM_STAGED=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_GEMV_WORKAROUNDS=YES
READY_FOR_PHASE14_ML_STORAGE_NUMERIC_CONVERSION_POLICY=YES
READY_FOR_TRITON=NO

Phase 13 locks GEMV-v0 row-wise dot as a composition, not a new target dot op:
`vector<16xf32>` elementwise multiply feeds finite-tree add reduction and then
a scalar store. The accepted TTIR form is real Triton-emitted
`tl.sum(a * x, axis=0)` over one vector K block, plus an independently stored
partial K-block result. Full cross-block K accumulation is staged.

The final hardware proof covers both value and TTIR paths. TTIR mixed
acceptance uses source-controlled real Triton sources and source-controlled
real TTIR snapshots, imports through the C++ TTIR importer, lowers through the
standard value layer, then through VC4Kernel, SSAVC4, scheduled VC4, and real
hardware with `active_qpus=12`, strict CPU oracles, sentinels, nonzero output
hashes, and zero mismatches.

No support is claimed for `tl.dot`, `tt.dot`, `vector.contract`, full GEMM,
atomics, cross-program accumulation, exact/default f32 dot, f16/subword dot, or
multi-block K accumulation into final `y[row]`. `READY_FOR_TRITON=NO` remains
the global gate.
