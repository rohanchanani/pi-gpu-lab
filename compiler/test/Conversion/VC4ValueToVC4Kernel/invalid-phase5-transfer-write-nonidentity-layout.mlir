// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @write_nonidentity_layout(
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
    %s: i32 {vc4value.arg_name = "s", vc4value.scalar_role = "stride"},
    %out: memref<?xf32, strided<[?]>, #vc4value.global> {vc4value.arg_name = "out",
                                                          vc4value.direction = "out",
                                                          vc4value.shape_args = ["n"],
                                                          vc4value.stride_args = ["s"]})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %value = arith.constant dense<1.000000e+00> : vector<16xf32>
  vector.transfer_write %value, %out[%c0] {in_bounds = [true]} : vector<16xf32>, memref<?xf32, strided<[?]>, #vc4value.global>
  return
}

// CHECK: expected rank-1 contiguous i32/f32/f16 #vc4value.global memref
// CHECK: not Phase 5 lowerable
// CHECK: staged value-surface feature
