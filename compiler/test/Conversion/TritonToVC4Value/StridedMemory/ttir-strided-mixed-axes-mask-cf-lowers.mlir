// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase11_strided_ranked_memory/generated/mixed_ttir_strided_memory_axes_mask_cf_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s

// CHECK-LABEL: func.func @mixed_ttir_strided_memory_axes_mask_cf_b16_kernel
// CHECK-SAME: memref<?xf32, #vc4value.global>
// CHECK-SAME: vc4value.grid_rank = 2
// CHECK: vc4value.program_id
// CHECK-SAME: axis = 0
// CHECK: vc4value.program_id
// CHECK-SAME: axis = 1
// CHECK: arith.muli
// CHECK: arith.index_cast
// CHECK: arith.addi
// CHECK-SAME: : index
// CHECK: vector.create_mask
// CHECK: vector.transfer_read
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
// CHECK-NOT: vector<16xindex>
