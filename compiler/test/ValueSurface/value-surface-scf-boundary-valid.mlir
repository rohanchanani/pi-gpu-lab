// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

func.func @scf_boundary(
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %acc0 = vector.broadcast %zero : f32 to vector<16xf32>
  %step = vector.broadcast %one : f32 to vector<16xf32>
  // CHECK: scf.for
  // Surface-admissible only: executable lowering requires scf-to-cf canonicalization.
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %acc0)
      -> (vector<16xf32>) {
    %next = arith.addf %acc, %step : vector<16xf32>
    scf.yield %next : vector<16xf32>
  }
  %keep = arith.addf %result, %acc0 : vector<16xf32>
  return
}
