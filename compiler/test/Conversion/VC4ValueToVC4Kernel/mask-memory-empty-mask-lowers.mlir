// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @mask_memory_empty(
    %in: memref<64xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %out: memref<64xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %base: index {vc4value.arg_name = "base"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.000000e+00 : f32
  %mask = vector.create_mask %c0 : vector<16xi1>
  %v = vector.transfer_read %in[%base], %zero, %mask {in_bounds = [true]} : memref<64xf32, #vc4value.global>, vector<16xf32>
  vector.transfer_write %v, %out[%base], %mask {in_bounds = [true]} : vector<16xf32>, memref<64xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @mask_memory_empty
// CHECK: vc4kernel.pred.empty
// CHECK-NOT: vc4kernel.pred.tail
// CHECK: vc4kernel.tmu_load_fragment
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
