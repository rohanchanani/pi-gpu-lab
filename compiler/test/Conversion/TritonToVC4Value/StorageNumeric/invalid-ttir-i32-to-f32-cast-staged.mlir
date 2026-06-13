// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase14_ml_storage_numeric/generated/ttir_i32_to_f32_cast_b16.ttir.mlir" --convert-triton-to-vc4-value -o - 2>&1 | FileCheck %s

// CHECK: i32 to f32 numeric cast staged by lower-half gap
// CHECK: READY_FOR_TRITON remains NO
