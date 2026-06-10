// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @compute_mask_select_not_memory_mask(
    %in: memref<64xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %out: memref<64xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %base: index {vc4value.arg_name = "base"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite"} {
  %zero = arith.constant 0.000000e+00 : f32
  %zero_v = arith.constant dense<0.000000e+00> : vector<16xf32>
  %tail = vector.create_mask %n : vector<16xi1>
  %x = vector.transfer_read %in[%base], %zero, %tail {in_bounds = [true]} : memref<64xf32, #vc4value.global>, vector<16xf32>
  %compute = arith.cmpf ogt, %x, %zero_v : vector<16xf32>
  %selected = arith.select %compute, %x, %zero_v : vector<16xi1>, vector<16xf32>
  vector.transfer_write %selected, %out[%base], %tail {in_bounds = [true]} : vector<16xf32>, memref<64xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @compute_mask_select_not_memory_mask
// CHECK: vc4kernel.pred.tail
// CHECK: vc4kernel.tmu_load_fragment
// CHECK: vc4kernel.fragment_cmp
// CHECK-SAME: predicate = #vc4kernel.cmp<ogt>
// CHECK: vc4kernel.fragment_select
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: sparse or unknown transfer
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
