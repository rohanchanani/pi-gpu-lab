// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @scalar_to_vector_f32_broadcast_lowers(
    %out: memref<16xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %alpha: f32 {vc4value.arg_name = "alpha"},
    %idx: index {vc4value.arg_name = "idx"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %v = vector.broadcast %alpha : f32 to vector<16xf32>
  vector.transfer_write %v, %out[%idx] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @scalar_to_vector_f32_broadcast_lowers
// CHECK: vc4kernel.splat {{.*}} : f32 -> vector<16xf32>
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: vector.
// CHECK-NOT: memref.
