PHASE15_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_APPROX_MATH_SFU_SOFTMAX
VALUE_APPROX_MATH_SFU_CONTRACT=LOCKED
VALUE_APPROX_SFU_STATIC=PASS
VALUE_APPROX_SFU_EXP_STATIC=PASS
VALUE_APPROX_SFU_RECIP_DIV_STATIC=PASS
VALUE_FINITE_F32_MAX_REDUCTION_STATIC=PASS
VALUE_SOFTMAX_V0_STATIC=PASS
VALUE_APPROX_SFU_HARDWARE_ISOLATION=PASS
VALUE_SOFTMAX_V0_HARDWARE_ISOLATION=PASS
VALUE_APPROX_SFU_SOFTMAX_MIXED_ACCEPTANCE=PASS
REAL_TRITON_SFU_SOFTMAX_SOURCES=YES
REAL_TTIR_SFU_SOFTMAX_SNAPSHOTS=YES
TTIR_APPROX_SFU_IMPORTER_STATIC=PASS
TTIR_SOFTMAX_V0_IMPORTER_STATIC=PASS
TTIR_APPROX_SFU_SOFTMAX_HARDWARE_ISOLATION=PASS
TTIR_APPROX_SFU_SOFTMAX_MIXED_ACCEPTANCE=PASS
APPROX_MATH_POLICY=EXPLICIT
VALUE_APPROX_SFU_TOLERANCE_POLICY=LOCKED
EXACT_DEFAULT_MATH_REJECTED=YES
ZERO_ACTIVE_SOFTMAX_STATUS=STAGED_OR_EXPLICIT_NOOP_GUARD_PROVEN
MULTIBLOCK_SOFTMAX_STAGED=YES
FULL_ATTENTION_STAGED=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_SFU_SOFTMAX_WORKAROUNDS=YES
READY_FOR_PHASE16_ATTENTION_SCORE_SOFTMAX_APPLY=YES
READY_FOR_TRITON=NO

# VC4 Vector/Triton Phase 15 SFU/Softmax Final Lock

Phase 15 locks base-2 SFU-backed approximate public natural math and one-block
softmax v0 for both the value layer and controlled real TTIR snapshots.

The locked value subset includes explicit approximate-SFU policy, natural
`math.exp` through `EXP2_X_LOG2E`, natural `math.log` through `LOG2_X_LN2`,
sqrt through `RSQRT_TIMES_X`, approximate reciprocal/division, finite f32 max
reduction, scalar-to-vector f32 broadcast, and the stable one-block softmax v0
composite. Exact/default math and exact NaN/Inf semantics remain rejected.

The locked TTIR subset includes controlled real Triton-emitted snapshots for
`tl.exp`, reciprocal/division, finite `tl.max`, stable softmax v0, and the
mixed f16-storage softmax fixture. Accepted TTIR snapshots enter through the
C++ importer to the standard value layer and then lower through VC4Kernel,
SSAVC4, scheduled VC4, and real hardware.

Final hardware evidence includes value isolation, value mixed acceptance, TTIR
isolation, and TTIR mixed acceptance with `active_qpus=12`, strict host oracles,
sentinels, nonzero output hashes, zero output mismatches, zero sentinel
mismatches, and zero launch failures.

Still staged: exact/default math, zero-active softmax without a proven finite
no-op guard, multiblock softmax, full attention, FlashAttention, `tt.dot`,
`vector.contract`, and GEMM. Global `READY_FOR_TRITON` remains `NO`.
