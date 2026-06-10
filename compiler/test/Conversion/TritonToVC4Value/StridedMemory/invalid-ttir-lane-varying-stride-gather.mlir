// RUN: not %vc4_triton_opt "%vc4_repo_root/examples/triton/phase11_strided_ranked_memory/generated/ttir_lane_varying_stride_gather_reject_b16.ttir.mlir" --convert-triton-to-vc4-value -o %t 2>&1 | FileCheck %s

// CHECK: lane-varying stride/gather pointer expression staged
// CHECK: READY_FOR_TRITON remains NO
