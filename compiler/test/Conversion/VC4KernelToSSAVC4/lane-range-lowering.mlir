// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s
// CHECK-LABEL: ssavc4.func @lanes
// CHECK: ssavc4.element_number : vector<16xi32>
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @lanes attributes {
    public_name = "lanes",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    resource = {
      uses_vpm = false, uses_barrier = false,
      require_full_block_residency = false,
      warps_per_block_max = 1 : i32, vpm_rows_per_block = 0 : i32,
      vpm_bytes_per_block = 0 : i32, semaphores_per_block = 0 : i32
    }
  } {
    %r = vc4kernel.lane_range : vector<16xi32>
    vc4kernel.return
  }
}
