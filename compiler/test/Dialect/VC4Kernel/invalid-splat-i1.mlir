// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %b = arith.constant true
    // CHECK: i32
    // CHECK-SAME: f32
    %v = vc4kernel.splat %b : i1 -> vector<16xi32>
    vc4kernel.return
  }
}
