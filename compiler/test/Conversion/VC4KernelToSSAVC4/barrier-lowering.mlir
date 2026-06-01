// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s
// CHECK-LABEL: ssavc4.func @bar
// CHECK-SAME: schedule_mode = "cooperative_block"
// CHECK: ssavc4.barrier
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @bar attributes {
    public_name = "bar",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    vc4kernel.barrier
    vc4kernel.return
  }
}
