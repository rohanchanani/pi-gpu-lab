// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @write_f32_full(%out: memref<64xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
                          %base: index {vc4value.arg_name = "base"},
                          %alpha: f32 {vc4value.arg_name = "alpha"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %value = vector.broadcast %alpha : f32 to vector<16xf32>
  vector.transfer_write %value, %out[%base] {in_bounds = [true]} : vector<16xf32>, memref<64xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @write_f32_full
// CHECK: vc4kernel.splat {{.*}} : f32 -> vector<16xf32>
// CHECK: vc4kernel.pred.full
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-SAME: coherency = #vc4kernel.coherency<dma_ordered>
// CHECK-SAME: inactive_store = #vc4kernel.inactive_store<preserve>
// CHECK-SAME: memory_path = #vc4kernel.memory_path<vdw_global_store>
// CHECK-SAME: : i32, vector<16xi32>, vector<16xf32>, <16>
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
// CHECK-NOT: linalg.
// CHECK-NOT: gpu.
// CHECK-NOT: tt.
