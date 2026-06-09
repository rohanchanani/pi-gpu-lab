// RUN: vc4-opt %s --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @scf_if_canonicalizes_to_cf_and_lowers(
    %out: memref<64xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %base: index {vc4value.arg_name = "base"},
    %n: index {vc4value.arg_name = "n"},
    %flag: i32 {vc4value.arg_name = "flag"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : i32
  %cond = arith.cmpi ne, %flag, %c0 : i32
  scf.if %cond {
    %value = arith.constant dense<1.000000e+00> : vector<16xf32>
    %mask = vector.create_mask %n : vector<16xi1>
    vector.transfer_write %value, %out[%base], %mask {in_bounds = [true]} : vector<16xf32>, memref<64xf32, #vc4value.global>
  }
  return
}

// CHECK-LABEL: vc4kernel.kernel @scf_if_canonicalizes_to_cf_and_lowers
// CHECK: cf.cond_br
// CHECK: vc4kernel.vdw_store_fragment
// CHECK: vc4kernel.return
// CHECK-NOT: scf.
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
