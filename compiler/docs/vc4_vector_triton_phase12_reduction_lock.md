PHASE12_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_REDUCTIONS
VALUE_REDUCTION_CONTRACT=LOCKED
VALUE_REDUCTION_TO_VC4KERNEL_STATIC=PASS
VALUE_VECTOR_REDUCTION_ADD_I32_STATIC=PASS
VALUE_VECTOR_REDUCTION_ADD_F32_FINITE_STATIC=PASS
VALUE_SCALAR_STORE_FOR_REDUCTION_STATIC=PASS
VALUE_REDUCTION_HARDWARE_ISOLATION=PASS
VALUE_REDUCTION_MIXED_ACCEPTANCE=PASS
REAL_TRITON_REDUCTION_SOURCES=YES
REAL_TTIR_REDUCTION_SNAPSHOTS=YES
TTIR_REDUCTION_IMPORTER_STATIC=PASS
TTIR_REDUCTION_HARDWARE_ISOLATION=PASS
TTIR_REDUCTION_MIXED_ACCEPTANCE=PASS
F32_REDUCTION_FINITE_TREE_POLICY=YES
EXACT_F32_REDUCTION_NOT_CLAIMED=YES
NON_ADD_REDUCTIONS_STAGED=YES
RANK_GT_1_REDUCTIONS_STAGED=YES
DOT_GEMV_STAGED_FOR_PHASE13=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_REDUCTION_WORKAROUNDS=YES
READY_FOR_PHASE13_GEMV_ROWWISE_DOT=YES
READY_FOR_TRITON=NO

# Phase 12 Reduction Final Lock

Phase 12 locks the value and controlled TTIR reduction subset. The accepted
forms are i32 add reductions over `vector<16xi32>`, f32 add reductions over
`vector<16xf32>` only under explicit finite-input and finite-tree policy, and
scalar rank-1 reduction-output stores. Tail reductions are represented by
Phase 10 inactive-zero input lanes before the unmasked reduction. Phase 11
row-strided memory may feed accepted row-slice reductions.

The TTIR bridge accepts source-controlled real Triton `tl.sum`/`tt.reduce` add
snapshots through the C++ importer into the standard value surface. Hardware
proof covers value isolation, value mixed acceptance, TTIR isolation, and TTIR
mixed acceptance with strict CPU oracles, sentinels, output hashes, and
`active_qpus=12` where required.

Phase 12 does not claim exact/default IEEE f32 reduction ordering. It also
does not claim max, min, product, custom reductions, rank>1 reductions,
`vector.multi_reduction`, scan, atomics, dot, GEMV, GEMM, f16/subword
reductions, or generalized math. Those forms remain staged for later phases.

No hardware fixture was weakened for the final lock. The Phase 12.9 TTIR mixed
fixture exposed a TTIR importer scalar pointer-offset composition bug; the fix
was made in the importer by preserving base scalar offset terms structurally
through `tt.addptr` chains. No lower-half workaround, fixture reshape, oracle
change, sentinel weakening, timeout weakening, or active-QPU reduction was
used.
