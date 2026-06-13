// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase14_ml_storage_numeric/generated/ttir_f32_to_i32_cast_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o - 2>&1 | FileCheck %s

// CHECK: fp-to-int numeric cast is staged
// CHECK: READY_FOR_TRITON remains NO
