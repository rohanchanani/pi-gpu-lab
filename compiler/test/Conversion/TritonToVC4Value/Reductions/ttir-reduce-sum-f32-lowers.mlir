// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase12_reductions/generated/ttir_reduce_sum_f32_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s

// CHECK-LABEL: func.func @ttir_reduce_sum_f32_b16_kernel
// CHECK-SAME: vc4value.fp_domain = "finite"
// CHECK-SAME: vc4value.reduction_policy = "finite_tree"
// CHECK: vector.transfer_read
// CHECK-SAME: vector<16xf32>
// CHECK: vector.reduction <add>
// CHECK-SAME: vector<16xf32> into f32
// CHECK: memref.store
// CHECK-NOT: tt.
// CHECK-NOT: ttg.
// CHECK-NOT: vc4kernel.
// CHECK-NOT: ssavc4.
