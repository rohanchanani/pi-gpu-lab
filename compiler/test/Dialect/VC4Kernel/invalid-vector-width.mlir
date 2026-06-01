// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    // CHECK: vector<16xi32>
    %v = vc4kernel.splat %c0 : i32 -> vector<8xi32>
    vc4kernel.return
  }
}
