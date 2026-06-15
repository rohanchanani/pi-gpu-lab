// REQUIRES: vc4-triton-cpp-frontend

// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase16_attention_apply_v0/generated/ttir_attention_apply_nontransposed_v_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t 2>&1 | FileCheck %s

// CHECK: lane-varying stride/gather pointer expression staged
// CHECK: READY_FOR_TRITON remains NO
