// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @mask_memory_full_unmasked(
    %in: memref<64xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %out: memref<64xi32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %base: index {vc4value.arg_name = "base"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %zero = arith.constant 0 : i32
  %v = vector.transfer_read %in[%base], %zero {in_bounds = [true]} : memref<64xi32, #vc4value.global>, vector<16xi32>
  vector.transfer_write %v, %out[%base] {in_bounds = [true]} : vector<16xi32>, memref<64xi32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @mask_memory_full_unmasked
// CHECK: vc4kernel.pred.full
// CHECK: vc4kernel.tmu_load_fragment
// CHECK-SAME: inactive_load = #vc4kernel.inactive_load<zero>
// CHECK: vc4kernel.pred.full
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-SAME: inactive_store = #vc4kernel.inactive_store<preserve>
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
