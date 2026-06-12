PHASE14_CONTROLLED_TRITON_STORAGE_NUMERIC_FIXTURES=YES
PHASE14_VALUE_ML_STORAGE_NUMERIC_CONTRACT=LOCKED
REAL_TRITON_ML_STORAGE_NUMERIC_SOURCES=YES
REAL_TTIR_ML_STORAGE_NUMERIC_SNAPSHOTS=YES
ACCEPTED_FIXTURES_EXCLUDE_UNRELATED_STAGED_FEATURES=YES
VALUE_F16_STORAGE_TO_F32_COMPUTE_SURFACE=ACCEPTED
VALUE_F32_COMPUTE_TO_F16_STORAGE_SURFACE=ACCEPTED
VALUE_F16_STORAGE_F32_COMPUTE_STATIC=PASS
VALUE_F16_STORE_F32_COMPUTE_STATIC=PASS
VALUE_F16_GEMV_INPUT_STORAGE_STATIC=PASS
VALUE_F16_STORAGE_F32_COMPUTE_HARDWARE_ISOLATION=PASS
VALUE_F16_LOAD_F32_COMPUTE_HARDWARE=PASS
VALUE_F32_COMPUTE_F16_STORE_HARDWARE=PASS
VALUE_F16_GEMV_INPUT_STORAGE_HARDWARE=PASS
VALUE_I32_TO_F32_CAST_HARDWARE=NOT_REQUIRED_STAGED_BY_LOWER_HALF_GAP
VALUE_F16_STORAGE_F32_COMPUTE_MIXED_ACCEPTANCE=PASS
VALUE_MIXED_REGRESSION=PASS
F16_STORAGE_FINITE_POLICY=YES
I32_TO_F32_CAST_STATUS=STAGED_BY_LOWER_HALF_GAP
NATIVE_F16_ARITHMETIC_STAGED=YES
BF16_FP8_STAGED=YES
INT8_INT16_QUANTIZED_STORAGE_STAGED=YES
NATIVE_F16_ARITHMETIC_FIXTURE_STAGED=YES
FP_TO_INT_CAST_FIXTURE_STAGED=YES
BF16_FP8_FIXTURE_STAGED_OR_NOT_GENERATED=YES
INT8_INT16_QUANT_STORAGE_STAGED_OR_NOT_GENERATED=YES
READY_FOR_PHASE14_3_VALUE_SURFACE_CONTRACT=YES
READY_FOR_PHASE14_4_VALUE_STORAGE_NUMERIC_STATIC=YES
READY_FOR_PHASE14_5_VALUE_HARDWARE_ISOLATION=YES
READY_FOR_PHASE14_6_VALUE_MIXED_ACCEPTANCE=YES
READY_FOR_PHASE14_7_TTIR_IMPORTER_STORAGE_NUMERIC_STATIC=YES
READY_FOR_TRITON=NO

# Phase 14 ML Storage/Numeric Controlled Fixtures

Phase 14.2 adds source-controlled real Triton sources and emitted TTIR
snapshots for ML storage and narrow numeric-conversion policy. These snapshots
define the Phase 14 TTIR fixture contract; they do not implement compiler
lowering and do not claim hardware support.

Accepted lowerable fixture contract:

- f16 storage loads represented by f16 pointer arguments and typed f16
  `tt.load`;
- f16 storage values promoted by `arith.extf` to f32 compute;
- f32 compute results stored to f16 through explicit `arith.truncf` before a
  typed f16 `tt.store`;
- Phase 13-style row-wise dot with f16 storage inputs, f32 multiply, f32 add
  reduction, and f32 scalar stores;
- mixed fixture combining multi-axis launch, tail masks, row-strided f16
  storage, control flow, f32 reduction, and scalar stores.

Accepted fixtures deliberately exclude unrelated staged features:

- no `tl.dot`, `tt.dot`, or `vector.contract`;
- no approximate math, SFU policy, softmax, atomics, block pointers, or
  gather/scatter;
- no `arith.sitofp`, `arith.fptosi`, bf16, or int8 forms in accepted
  snapshots;
- no native f16 arithmetic in accepted snapshots.

Staged fixture contract:

- `ttir_i32_to_f32_cast_b16` emits `arith.sitofp` and is staged by the current
  lower-half exact numeric-cast gap.
- `ttir_native_f16_arithmetic_reject_b16` emits `arith.addf` on
  `tensor<16xf16>` and remains staged.
- `ttir_f32_to_i32_cast_reject_b16` emits `arith.fptosi` and remains staged.
- `ttir_bf16_or_fp8_reject_b16` emits a clean bf16 pointer/load/promote form
  and remains staged/rejected by the locked target profile.
- `ttir_int8_quantized_storage_reject_b16` emits typed i8 load/store and i8
  arithmetic and remains staged for later quantized storage work.

Finite f16 storage policy:

The accepted f16 fixtures exercise finite f16 storage conversion with f32
compute. They do not claim native f16 arithmetic, exact/default math, exact
IEEE conversion semantics, or broad Triton readiness.

Source and snapshot location:

```text
examples/triton/phase14_ml_storage_numeric/sources/
examples/triton/phase14_ml_storage_numeric/generated/
examples/triton/phase14_ml_storage_numeric/manifest.json
```

Validation:

- every generated snapshot parses with
  `compiler/build-triton-llvm/bin/vc4-triton-opt`;
- `compiler/test/CodeGen/Triton/phase14-storage-numeric-snapshots-parse.test`
  audits the manifest and source-controlled snapshots;
- value import smoke is expected pending importer/value-surface repair in the
  following Phase 14 packages.

Phase 14.5 value hardware isolation:

- `value_f16_load_f32_compute_store_f32_vc4value` proves f16 storage loads
  extended to f32 compute and f32 stores on hardware.
- `value_f32_compute_store_f16_vc4value` proves f32 compute narrowed to f16
  storage with the explicit finite storage policy.
- `value_f16_row_dot_f32_accum_vc4value` proves f16 GEMV input storage with f32
  accumulation/output using the Phase 13 row-wise dot shape.
- `value_f16_empty_repeat_vc4value` proves empty and repeated launches preserve
  sentinels while exercising the f16 load-to-f32 path.
- i32-to-f32 hardware isolation is not required in this phase because the cast
  remains staged by the lower-half gap.

Phase 14.6 value mixed acceptance:

- `mixed_value_f16_storage_gemv_axes_mask_cf_reduction_vc4value` combines
  f16 storage load/store, f32 compute, GEMV row-wise partial dots, multi-axis
  launch, masks, control flow, row-strided memory, scalar reductions, and
  repeated mixed-suite regression coverage.
- The full VC4Value mixed acceptance suite passed with active_qpus=12 where
  applicable, zero mismatches, zero sentinel mismatches, and zero launch
  failures.

`READY_FOR_TRITON=NO` remains locked.
