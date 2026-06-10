// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase10_mask_memory/generated/ttir_mask_compute_select_tail_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s

// CHECK-LABEL: func.func @ttir_mask_compute_select_tail_b16_kernel
// CHECK-SAME: vc4value.fp_domain = "finite"
// CHECK: vector.create_mask
// CHECK: vector.transfer_read
// CHECK: arith.cmpf
// CHECK: arith.select
// CHECK: vector.transfer_write
// CHECK-NOT: tt.
// CHECK-NOT: ttg.
// CHECK-NOT: vc4kernel.
// CHECK-NOT: ssavc4.
// CHECK-NOT: vc4.module
