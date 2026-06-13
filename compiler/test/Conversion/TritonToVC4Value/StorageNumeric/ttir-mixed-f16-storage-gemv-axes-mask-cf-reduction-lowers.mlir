// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase14_ml_storage_numeric/generated/mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s --implicit-check-not='tt.' --implicit-check-not='ttg.' --implicit-check-not='triton_gpu.' --implicit-check-not='vc4kernel.' --implicit-check-not='ssavc4.'

// CHECK-LABEL: func.func @mixed_ttir_f16_storage_gemv_axes_mask_cf_reduction_b16
// CHECK-SAME: vc4value.fp_domain = "finite"
// CHECK-SAME: vc4value.grid_rank = 2
// CHECK-SAME: vc4value.reduction_policy = "finite_tree"
// CHECK: vc4value.program_id
// CHECK-SAME: axis = 0
// CHECK: vc4value.program_id
// CHECK-SAME: axis = 1
// CHECK: vector.transfer_read
// CHECK-SAME: memref<16xf16, #vc4value.global>, vector<16xf16>
// CHECK: arith.extf
// CHECK: vector.transfer_read
// CHECK-SAME: memref<16xf16, #vc4value.global>, vector<16xf16>
// CHECK: arith.extf
// CHECK: arith.mulf
// CHECK-SAME: vector<16xf32>
// CHECK: vector.reduction <add>
// CHECK: scf.if
// CHECK: memref.store
