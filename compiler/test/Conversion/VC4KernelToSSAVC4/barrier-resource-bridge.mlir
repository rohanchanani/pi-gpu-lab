// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @barrier_resource
// CHECK-SAME: builtins = [
// CHECK-SAME: logical_warp_id
// CHECK-SAME: warps_per_block
// CHECK-SAME: semaphore_base
// CHECK-SAME: vc4.resource = {
// CHECK-SAME: requires_semaphore_base_builtin = true
// CHECK-SAME: schedule_mode = "cooperative_block"
// CHECK-SAME: semaphore_count_per_block = 4 : i32
// CHECK-SAME: uses_barrier = true
// CHECK-SAME: warps_per_block = 4 : i32
// CHECK: %[[WARP:.*]] = ssavc4.uniform.read 0 : i32
// CHECK: %[[WARPS:.*]] = ssavc4.uniform.read 1 : i32
// CHECK: %[[SEMA:.*]] = ssavc4.uniform.read 2 : i32
// CHECK: ssavc4.barrier %[[WARP]], %[[WARPS]], %[[SEMA]] : i32, i32, i32
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @barrier_resource attributes {
    public_name = "barrier_resource",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [],
    warps_per_block = 4 : i32
  } {
    vc4kernel.barrier
    vc4kernel.return
  }
}
