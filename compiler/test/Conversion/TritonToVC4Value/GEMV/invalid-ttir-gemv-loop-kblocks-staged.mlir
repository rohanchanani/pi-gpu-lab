// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase13_gemv_rowwise_dot/generated/ttir_gemv_loop_kblocks_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t.mlir 2>&1 | FileCheck %s

// CHECK: multi-block K accumulation is staged
// CHECK: READY_FOR_TRITON remains NO
