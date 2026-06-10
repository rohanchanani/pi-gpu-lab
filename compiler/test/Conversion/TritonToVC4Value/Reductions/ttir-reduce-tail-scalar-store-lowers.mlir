// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase12_reductions/generated/ttir_reduce_sum_tail_scalar_store_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s

// CHECK-LABEL: func.func @ttir_reduce_sum_tail_scalar_store_b16_kernel
// CHECK-SAME: vc4value.reduction_policy = "finite_tree"
// CHECK: vector.create_mask
// CHECK: vector.transfer_read
// CHECK: vector.reduction <add>
// CHECK: arith.cmpi
// CHECK: scf.if
// CHECK: memref.store
// CHECK-NOT: vector.transfer_write
// CHECK-NOT: tt.
// CHECK-NOT: ttg.
// CHECK-NOT: vc4kernel.
// CHECK-NOT: ssavc4.
