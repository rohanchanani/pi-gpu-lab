// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @scalar_memref_store_reduction_output_lowers(
    %out: memref<16xi32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %value: i32 {vc4value.arg_name = "value"},
    %idx: index {vc4value.arg_name = "idx"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  memref.store %value, %out[%idx] : memref<16xi32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @scalar_memref_store_reduction_output_lowers
// CHECK: %[[SPLAT:.*]] = vc4kernel.splat
// CHECK: %[[LANE0:.*]] = vc4kernel.pred.tail
// CHECK: vc4kernel.vdw_store_fragment {{.*}}, {{.*}}, %[[SPLAT]], %[[LANE0]]
// CHECK-SAME: inactive_store = #vc4kernel.inactive_store<preserve>
// CHECK-NOT: vector.
// CHECK-NOT: memref.
