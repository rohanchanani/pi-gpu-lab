// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase13_gemv_rowwise_dot/generated/ttir_gemv_row_dot_i32_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t.value.mlir
// RUN: not %vc4_opt %t.value.mlir --vc4-verify-value-surface --convert-scf-to-cf --convert-vc4-value-to-vc4kernel --verify-vc4kernel -o %t.vc4kernel.mlir 2>&1 | FileCheck %s

// CHECK: vector i32 muli requires vc4value.i32_mul_policy = "mul24_safe"
// CHECK: READY_FOR_TRITON remains NO
