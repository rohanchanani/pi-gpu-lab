PHASE14_RESULT=LOCKED
FEATURE=VALUE_AND_TTIR_ML_STORAGE_NUMERIC_CONVERSION_POLICY
VALUE_ML_STORAGE_NUMERIC_CONTRACT=LOCKED
VALUE_F16_STORAGE_F32_COMPUTE_STATIC=PASS
VALUE_F16_STORAGE_F32_COMPUTE_HARDWARE_ISOLATION=PASS
VALUE_F16_STORAGE_F32_COMPUTE_MIXED_ACCEPTANCE=PASS
REAL_TRITON_ML_STORAGE_NUMERIC_SOURCES=YES
REAL_TTIR_ML_STORAGE_NUMERIC_SNAPSHOTS=YES
TTIR_F16_STORAGE_IMPORTER_STATIC=PASS
TTIR_F16_STORAGE_HARDWARE_ISOLATION=PASS
TTIR_F16_STORAGE_MIXED_ACCEPTANCE=PASS
F16_STORAGE_FINITE_POLICY=YES
I32_TO_F32_CAST_STATUS=STAGED_BY_LOWER_HALF_GAP
NATIVE_F16_ARITHMETIC_STAGED=YES
BF16_FP8_STAGED=YES
INT8_INT16_QUANTIZED_STORAGE_STAGED=YES
FP_TO_INT_CASTS_STAGED=YES
EXACT_UNPOLICY_NUMERIC_CASTS_STAGED=YES
SOFTMAX_SFU_STAGED_FOR_PHASE15=YES
VALUE_MIXED_REGRESSION=PASS
TTIR_MIXED_REGRESSION=PASS
LAYERED_REGRESSION_POLICY_APPLIED=YES
NO_TEMPORARY_STORAGE_NUMERIC_WORKAROUNDS=YES
READY_FOR_PHASE15_APPROX_MATH_SFU_SOFTMAX=YES
READY_FOR_TRITON=NO

# Phase 14 ML Storage Numeric Conversion Policy Lock

Phase 14 locks the value and TTIR f16-storage/f32-compute subset through real
Triton sources, source-controlled TTIR snapshots, C++ TTIR import, standard
value lowering, VC4Kernel, SSAVC4, scheduled VC4, and hardware.

The accepted executable policy is narrow:

- f16 storage loads convert to f32 compute values;
- f32 compute values store to f16 storage only under the explicit finite f16
  storage policy;
- f16 row-wise dot input storage accumulates in f32 using the Phase 13 GEMV
  shape;
- i32-to-f32 casts remain staged by the lower-half gap.

The final lock does not claim native f16 arithmetic, bf16/fp8 conversion or
arithmetic, int8/int16 quantized storage, fp-to-int casts, exact/unpolicy
numeric casts, softmax/SFU, `tt.dot`, `vector.contract`, or GEMM.

The final layered mixed regressions passed for both VC4Value and TTIR. Phase 15
may start the approximate math/SFU/softmax package while global Triton
readiness remains locked off.
