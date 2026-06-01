// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  vc4kernel.kernel @barrier attributes {
    public_name = "barrier",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    // CHECK: schedule_mode = #vc4kernel.schedule_mode<cooperative_block>
    // CHECK: vc4kernel.barrier
    vc4kernel.barrier
    vc4kernel.return
  }
}
