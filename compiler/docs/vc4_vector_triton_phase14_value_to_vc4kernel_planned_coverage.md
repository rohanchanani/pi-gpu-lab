PHASE14_VALUE_TO_VC4KERNEL_PLANNED_COVERAGE=YES
PHASE14_VALUE_ML_STORAGE_NUMERIC_CONTRACT=LOCKED
VALUE_F16_STORAGE_TO_F32_COMPUTE_SURFACE=ACCEPTED
VALUE_F32_COMPUTE_TO_F16_STORAGE_SURFACE=ACCEPTED
F16_STORAGE_FINITE_POLICY=YES
I32_TO_F32_CAST_STATUS=STAGED_BY_LOWER_HALF_GAP
NATIVE_F16_ARITHMETIC_STAGED=YES
BF16_FP8_STAGED=YES
INT8_INT16_QUANTIZED_STORAGE_STAGED=YES
READY_FOR_PHASE14_4_VALUE_STORAGE_NUMERIC_STATIC=YES
READY_FOR_TRITON=NO

# Phase 14 Value-To-VC4Kernel Planned Coverage

This document records the static coverage that Phase 14.4 must prove before any
Phase 14 hardware package. It is a planning contract only and does not implement
or claim executable lowering.

Accepted planned coverage:

- `f16_transfer_read_extf_f32_compute`: f16 `vector.transfer_read` from
  Phase 10 rank-1 identity memory, zero inactive lanes, `arith.extf` to
  `vector<16xf32>`, and existing f32 compute lowering.
- `f32_compute_truncf_f16_transfer_write`: f32 compute,
  `arith.truncf vector<16xf32> -> vector<16xf16>`, and f16
  `vector.transfer_write` with explicit
  `vc4value.f16_storage_policy = "finite"`.
- `f16_row_strided_storage`: Phase 11 row-strided memory with f16 element type
  and contiguous `vector<16>` lane transfers.
- `f16_row_dot_input_storage`: Phase 13 row-wise dot with f16 A/X storage
  inputs, f32 multiply, finite-tree f32 reduction, and scalar f32 output store.

Required reject/staged coverage:

- f16 store missing `vc4value.f16_storage_policy = "finite"` rejects.
- native f16 arithmetic rejects.
- f16 reduction and f16 accumulation reject.
- `arith.fptosi` and `arith.fptoui` reject as staged fp-to-int casts.
- bf16/fp8 native conversion or arithmetic rejects.
- int8/int16 quantized storage, scale, zero-point, and narrow quantized
  arithmetic policy remain staged.
- `arith.sitofp`/`arith.uitofp` i32/index-to-f32 casts remain
  `STAGED_BY_LOWER_HALF_GAP` unless Phase 14.4 proves a real lower-half path.

Out of scope:

- native f16 arithmetic;
- bf16/fp8 support;
- int8/int16 quantization;
- approximate math/SFU/softmax;
- `tt.dot`, `vector.contract`, and GEMM;
- broad exact/default numeric conversion support.

`READY_FOR_TRITON=NO` remains locked.
