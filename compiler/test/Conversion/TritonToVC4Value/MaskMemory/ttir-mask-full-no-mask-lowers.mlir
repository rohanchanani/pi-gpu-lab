// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase10_mask_memory/generated/ttir_mask_full_no_mask_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s

// CHECK-LABEL: func.func @ttir_mask_full_no_mask_b16_kernel
// CHECK-SAME: memref<16xf32, #vc4value.global>
// CHECK: vc4value.program_id
// CHECK-NOT: vector.create_mask
// CHECK: vector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<16xf32, #vc4value.global>, vector<16xf32>
// CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}] : vector<16xf32>, memref<16xf32, #vc4value.global>
// CHECK-NOT: tt.
// CHECK-NOT: ttg.
// CHECK-NOT: vc4kernel.
// CHECK-NOT: ssavc4.
// CHECK-NOT: vc4.module
