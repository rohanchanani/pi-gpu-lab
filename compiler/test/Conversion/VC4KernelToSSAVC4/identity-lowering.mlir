// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s
// CHECK-LABEL: ssavc4.func @ids
// CHECK-SAME: builtins = [
// CHECK-SAME: logical_request
// CHECK-SAME: logical_warp_id
// CHECK: ssavc4.uniform.read
// CHECK: ssavc4.uniform.read
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @ids attributes {
    public_name = "ids",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    %p = vc4kernel.program_id {axis = 0 : i32} : i32
    %w = vc4kernel.warp_id : i32
    vc4kernel.return
  }
}
