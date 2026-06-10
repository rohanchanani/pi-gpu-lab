// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase12_reductions/generated/mixed_ttir_reduction_axes_mask_cf_strided_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s

// CHECK-LABEL: func.func @mixed_ttir_reduction_axes_mask_cf_strided_b16_kernel
// CHECK-SAME: vc4value.grid_rank = 2
// CHECK-SAME: vc4value.reduction_policy = "finite_tree"
// CHECK: vc4value.program_id
// CHECK-SAME: axis = 0
// CHECK: vc4value.program_id
// CHECK-SAME: axis = 1
// CHECK: vc4value.num_programs
// CHECK: vector.create_mask
// CHECK: vector.transfer_read
// CHECK: vector.reduction <add>
// CHECK: scf.if
// CHECK: [[ROW_BLOCKS:%.*]] = arith.muli
// CHECK: [[ROW_INDEX:%.*]] = arith.index_cast [[ROW_BLOCKS]]
// CHECK: [[BLOCK_INDEX:%.*]] = arith.index_cast
// CHECK: [[STORE_INDEX:%.*]] = arith.addi [[ROW_INDEX]], [[BLOCK_INDEX]]
// CHECK: memref.store {{%.*}}, {{%.*}}[[STORE_INDEX]]
// CHECK-NOT: tt.
// CHECK-NOT: ttg.
// CHECK-NOT: vc4kernel.
// CHECK-NOT: ssavc4.
