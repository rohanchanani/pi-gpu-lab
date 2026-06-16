// REQUIRES: vc4-triton-cpp-frontend

// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase17_online_softmax_state/generated/ttir_online_attention_nontransposed_v_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t 2>&1 | FileCheck %s

// CHECK: non-contiguous column slice pointer expression staged
// CHECK: READY_FOR_TRITON remains NO
