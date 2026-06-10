// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @reduction_i32_add_lowers(
    %in: memref<64xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %out: memref<16xi32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %base: index {vc4value.arg_name = "base"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %zero = arith.constant 0 : i32
  %v = vector.transfer_read %in[%base], %zero {in_bounds = [true]} : memref<64xi32, #vc4value.global>, vector<16xi32>
  %sum = vector.reduction <add>, %v : vector<16xi32> into i32
  memref.store %sum, %out[%base] : memref<16xi32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @reduction_i32_add_lowers
// CHECK: %[[LOAD:.*]] = vc4kernel.tmu_load_fragment
// CHECK: %[[RED:.*]] = vc4kernel.fragment_reduce %[[LOAD]], {{.*}} {kind = #vc4kernel.reduce<add>}
// CHECK: %[[LANE0:.*]] = vc4kernel.pred.tail
// CHECK: vc4kernel.vdw_store_fragment {{.*}}, {{.*}}, %[[RED]], %[[LANE0]]
// CHECK-SAME: inactive_store = #vc4kernel.inactive_store<preserve>
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
