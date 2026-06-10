// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @reduction_under_cf_guard_lowers(
    %in: memref<64xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %out: memref<16xi32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %base: index {vc4value.arg_name = "base"},
    %flag: i32 {vc4value.arg_name = "flag"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %zero = arith.constant 0 : i32
  %v = vector.transfer_read %in[%base], %zero {in_bounds = [true]} : memref<64xi32, #vc4value.global>, vector<16xi32>
  %sum = vector.reduction <add>, %v : vector<16xi32> into i32
  %do_store = arith.cmpi ne, %flag, %zero : i32
  cf.cond_br %do_store, ^store, ^exit
^store:
  memref.store %sum, %out[%base] : memref<16xi32, #vc4value.global>
  cf.br ^exit
^exit:
  return
}

// CHECK-LABEL: vc4kernel.kernel @reduction_under_cf_guard_lowers
// CHECK: vc4kernel.fragment_reduce
// CHECK: cf.cond_br
// CHECK: vc4kernel.vdw_store_fragment
// CHECK: cf.br
// CHECK-NOT: vector.
// CHECK-NOT: memref.
