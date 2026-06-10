// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @atomic_reduction(
      %out: memref<?xi32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "inout", vc4value.shape_args = ["n"]},
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : i32
    // expected-error @+1 {{direct memref side-effect operation is not legal in the VC4 value surface; use structured vector transfer/planning forms}}
    %old = memref.atomic_rmw addi %one, %out[%c0] : (i32, memref<?xi32, #vc4value.global>) -> i32
    return
  }
}
