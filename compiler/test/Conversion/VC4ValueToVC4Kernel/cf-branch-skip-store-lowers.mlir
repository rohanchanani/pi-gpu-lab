// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @cf_branch_skip_store(
    %out: memref<64xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %base: index {vc4value.arg_name = "base"},
    %n: index {vc4value.arg_name = "n"},
    %do_store: i32 {vc4value.arg_name = "do_store"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : i32
  %one = arith.constant dense<1.000000e+00> : vector<16xf32>
  %mask = vector.create_mask %n : vector<16xi1>
  %cond = arith.cmpi ne, %do_store, %c0 : i32
  cf.cond_br %cond, ^store, ^exit

^store:
  vector.transfer_write %one, %out[%base], %mask {in_bounds = [true]} : vector<16xf32>, memref<64xf32, #vc4value.global>
  cf.br ^exit

^exit:
  return
}

// CHECK-LABEL: vc4kernel.kernel @cf_branch_skip_store
// CHECK: cf.cond_br
// CHECK: vc4kernel.vdw_store_fragment
// CHECK: cf.br
// CHECK: vc4kernel.return
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
// CHECK-NOT: linalg.
// CHECK-NOT: gpu.
// CHECK-NOT: tt.
