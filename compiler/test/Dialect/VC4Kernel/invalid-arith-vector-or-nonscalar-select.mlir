// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %cond = arith.constant true
    %lhs = arith.constant dense<1> : vector<16xi32>
    %rhs = arith.constant dense<0> : vector<16xi32>
    // CHECK: arith operations may not operate on or produce vectors
    %r = arith.select %cond, %lhs, %rhs : vector<16xi32>
    vc4kernel.return
  }
}
