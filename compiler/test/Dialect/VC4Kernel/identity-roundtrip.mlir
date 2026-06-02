// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  vc4kernel.kernel @ids attributes {
    public_name = "ids",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    // CHECK: vc4kernel.program_id {axis = 0 : i32}
    %p = vc4kernel.program_id {axis = 0 : i32} : i32
    // CHECK: vc4kernel.program_id {axis = 1 : i32}
    %py = vc4kernel.program_id {axis = 1 : i32} : i32
    // CHECK: vc4kernel.num_programs {axis = 2 : i32}
    %nz = vc4kernel.num_programs {axis = 2 : i32} : i32
    // CHECK: vc4kernel.warp_id
    %w = vc4kernel.warp_id : i32
    // CHECK: vc4kernel.lane_range
    %r = vc4kernel.lane_range : vector<16xi32>
    vc4kernel.return
  }
}
