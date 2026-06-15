// REQUIRES: vc4-triton-cpp-frontend

// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase16_attention_apply_v0/generated/ttir_attention_apply_k_zero_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t 2>&1 | FileCheck %s

// CHECK: sparse or unknown tt.load memory mask
// CHECK: READY_FOR_TRITON remains NO
