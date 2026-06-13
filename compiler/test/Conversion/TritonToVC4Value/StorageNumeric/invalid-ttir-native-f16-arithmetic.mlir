// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase14_ml_storage_numeric/generated/ttir_native_f16_arithmetic_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o - 2>&1 | FileCheck %s

// CHECK: native f16 arithmetic is staged
// CHECK: READY_FOR_TRITON remains NO
