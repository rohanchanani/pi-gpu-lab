// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase10_mask_memory/generated/ttir_mask_tail_load_store_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s

// CHECK-LABEL: func.func @ttir_mask_tail_load_store_b16_kernel
// CHECK-SAME: memref<?xf32, #vc4value.global>
// CHECK-SAME: %{{.*}}: index
// CHECK: vc4value.program_id
// CHECK: arith.subi
// CHECK: vector.create_mask
// CHECK: vector.transfer_read
// CHECK-SAME: vector<16xf32>
// CHECK: vector.transfer_write
// CHECK-SAME: vector<16xf32>
// CHECK-NOT: tt.
// CHECK-NOT: ttg.
// CHECK-NOT: vc4kernel.
// CHECK-NOT: ssavc4.
// CHECK-NOT: vc4.module
