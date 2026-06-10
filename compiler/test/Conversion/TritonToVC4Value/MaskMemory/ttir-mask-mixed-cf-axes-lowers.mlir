// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase10_mask_memory/generated/mixed_ttir_mask_memory_cf_axes_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s

// CHECK-LABEL: func.func @mixed_ttir_mask_memory_cf_axes_b16_kernel
// CHECK-SAME: vc4value.grid_rank = 2
// CHECK: vc4value.program_id
// CHECK: vc4value.program_id
// CHECK: vc4value.num_programs
// CHECK: vector.create_mask
// CHECK: vector.transfer_read
// CHECK: arith.cmpf
// CHECK: arith.select
// CHECK: scf.if
// CHECK: vector.transfer_write
// CHECK-NOT: tt.
// CHECK-NOT: ttg.
// CHECK-NOT: vc4kernel.
// CHECK-NOT: ssavc4.
// CHECK-NOT: vc4.module
