// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  vc4kernel.kernel @ids attributes {
    public_name = "ids",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    resource = {
      uses_vpm = false, uses_barrier = false,
      require_full_block_residency = false,
      warps_per_block_max = 1 : i32, vpm_rows_per_block = 0 : i32,
      vpm_bytes_per_block = 0 : i32, semaphores_per_block = 0 : i32
    }
  } {
    // CHECK: vc4kernel.program_id
    %p = vc4kernel.program_id : i32
    // CHECK: vc4kernel.block_id
    %b = vc4kernel.block_id : i32
    // CHECK: vc4kernel.warp_id
    %w = vc4kernel.warp_id : i32
    // CHECK: vc4kernel.lane_id
    %l = vc4kernel.lane_id : i32
    // CHECK: vc4kernel.lane_range
    %r = vc4kernel.lane_range : vector<16xi32>
    vc4kernel.return
  }
}
