// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @reduction_tail_inactive_zero_lowers(
    %in: memref<64xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %out: memref<16xi32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %base: index {vc4value.arg_name = "base"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %remaining = arith.subi %n, %base : index
  %mask = vector.create_mask %remaining : vector<16xi1>
  %zero = arith.constant 0 : i32
  %v = vector.transfer_read %in[%base], %zero, %mask {in_bounds = [true]} : memref<64xi32, #vc4value.global>, vector<16xi32>
  %sum = vector.reduction <add>, %v : vector<16xi32> into i32
  memref.store %sum, %out[%base] : memref<16xi32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @reduction_tail_inactive_zero_lowers
// CHECK: vc4kernel.pred.tail
// CHECK: vc4kernel.tmu_load_fragment
// CHECK-SAME: inactive_load = #vc4kernel.inactive_load<zero>
// CHECK: vc4kernel.fragment_reduce
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-SAME: inactive_store = #vc4kernel.inactive_store<preserve>
// CHECK-NOT: vector.
// CHECK-NOT: memref.
