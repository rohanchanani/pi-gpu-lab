// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase13_gemv_rowwise_dot/generated/mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s --implicit-check-not='tt.' --implicit-check-not='ttg.' --implicit-check-not='triton_gpu.' --implicit-check-not='vc4kernel.' --implicit-check-not='ssavc4.'

// CHECK-LABEL: func.func @mixed_ttir_gemv_row_dot_axes_mask_cf_strided_reduction_b16_kernel
// CHECK-SAME: vc4value.fp_domain = "finite"
// CHECK-SAME: vc4value.grid_rank = 2
// CHECK-SAME: vc4value.reduction_policy = "finite_tree"
// CHECK: vc4value.program_id
// CHECK-SAME: axis = 0
// CHECK: vc4value.program_id
// CHECK-SAME: axis = 1
// CHECK: vector.create_mask
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: arith.mulf
// CHECK: vector.reduction <add>
// CHECK: scf.if
// CHECK: arith.addf
// CHECK: scf.if
// CHECK: memref.store
