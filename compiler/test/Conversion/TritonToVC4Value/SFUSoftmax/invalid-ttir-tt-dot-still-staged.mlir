// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase15_sfu_softmax/generated/ttir_tl_dot_still_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o - 2>&1 | FileCheck %s

// CHECK: tt.dot / contract staged by body feature
// CHECK: READY_FOR_TRITON remains NO
