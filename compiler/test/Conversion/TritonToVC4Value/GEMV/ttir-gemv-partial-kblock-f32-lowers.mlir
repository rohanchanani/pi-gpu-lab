// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase13_gemv_rowwise_dot/generated/ttir_gemv_partial_kblock_f32_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s --implicit-check-not='tt.' --implicit-check-not='ttg.' --implicit-check-not='triton_gpu.' --implicit-check-not='vc4kernel.' --implicit-check-not='ssavc4.'

// CHECK-LABEL: func.func @ttir_gemv_partial_kblock_f32_b16_kernel
// CHECK-SAME: vc4value.grid_rank = 2
// CHECK-SAME: vc4value.reduction_policy = "finite_tree"
// CHECK: vc4value.program_id
// CHECK-SAME: axis = 0
// CHECK: vc4value.program_id
// CHECK-SAME: axis = 1
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: arith.mulf
// CHECK: vector.reduction <add>
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: memref.store
