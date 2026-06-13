// REQUIRES: vc4-has-triton-cpp-frontend
// RUN: %vc4_triton_opt "%vc4_repo_root/examples/triton/phase14_ml_storage_numeric/generated/ttir_f32_compute_store_f16_b16.ttir.mlir" --convert-triton-to-vc4-value -o - | FileCheck %s --implicit-check-not='tt.' --implicit-check-not='ttg.' --implicit-check-not='triton_gpu.' --implicit-check-not='vc4kernel.' --implicit-check-not='ssavc4.'

// CHECK-LABEL: func.func @ttir_f32_compute_store_f16_b16
// CHECK-SAME: memref<16xf32, #vc4value.global>
// CHECK-SAME: memref<16xf16, #vc4value.global>
// CHECK-SAME: vc4value.f16_storage_policy = "finite"
// CHECK: vector.transfer_read
// CHECK-SAME: memref<16xf32, #vc4value.global>, vector<16xf32>
// CHECK: arith.addf
// CHECK-SAME: vector<16xf32>
// CHECK: arith.truncf
// CHECK-SAME: vector<16xf32> to vector<16xf16>
// CHECK: vector.transfer_write
// CHECK-SAME: vector<16xf16>, memref<16xf16, #vc4value.global>
