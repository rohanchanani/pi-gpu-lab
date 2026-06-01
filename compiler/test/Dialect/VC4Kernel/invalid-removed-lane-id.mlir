// RUN: not vc4-opt %s --allow-unregistered-dialect --verify-vc4kernel 2>&1 | FileCheck %s
module {
  vc4kernel.kernel @bad attributes {
    public_name = "bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    // CHECK: unknown or forbidden vc4kernel operation
    %lane = "vc4kernel.lane_id"() : () -> i32
    vc4kernel.return
  }
}
