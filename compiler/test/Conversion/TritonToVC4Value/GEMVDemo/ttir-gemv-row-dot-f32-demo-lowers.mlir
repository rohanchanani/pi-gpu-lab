// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase13_gemv_demo/generated/gemv_row_dot_f32_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s

// CHECK-LABEL: func.func @gemv_row_dot_f32_b16_kernel
// CHECK-SAME: vc4value.fp_domain = "finite"
// CHECK-SAME: vc4value.grid_rank = 1
// CHECK-SAME: vc4value.reduction_policy = "finite_tree"
// CHECK: vc4value.program_id
// CHECK-SAME: axis = 0
// CHECK: [[ROW_STRIDE_I32:%.*]] = arith.muli
// CHECK: [[ROW_STRIDE_INDEX:%.*]] = arith.index_cast [[ROW_STRIDE_I32]]
// CHECK: [[A:%.*]] = vector.transfer_read {{%.*}}[[ROW_STRIDE_INDEX]]
// CHECK-SAME: memref<?xf32, #vc4value.global>, vector<16xf32>
// CHECK: [[X:%.*]] = vector.transfer_read {{%.*}}[%c0
// CHECK-SAME: memref<?xf32, #vc4value.global>, vector<16xf32>
// CHECK: [[PRODUCT:%.*]] = arith.mulf [[A]], [[X]] : vector<16xf32>
// CHECK: [[REDUCE:%.*]] = vector.reduction <add>, [[PRODUCT]] : vector<16xf32> into f32
// CHECK: memref.store [[REDUCE]]
// CHECK-NOT: tt.
// CHECK-NOT: ttg.
// CHECK-NOT: triton_gpu.
// CHECK-NOT: vc4kernel.
// CHECK-NOT: ssavc4.
// CHECK-NOT: unrealized_conversion_cast
