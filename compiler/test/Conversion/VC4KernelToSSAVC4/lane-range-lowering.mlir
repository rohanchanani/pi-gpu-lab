// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s
// CHECK-LABEL: ssavc4.func @lanes
// CHECK: ssavc4.element_number : vector<16xi32>
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @lanes attributes {
    public_name = "lanes",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %r = vc4kernel.lane_range : vector<16xi32>
    vc4kernel.return
  }
}
