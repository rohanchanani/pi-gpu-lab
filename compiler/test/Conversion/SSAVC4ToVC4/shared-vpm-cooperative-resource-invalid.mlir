// RUN: not vc4-opt %s --convert-ssavc4-to-vc4 -o /dev/null 2>&1 | FileCheck %s

ssavc4.module @bad_shared_vpm_resource {
  ssavc4.func @bad_shared_vpm_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      uses_barrier = false,
      uses_shared_vpm = false,
      require_full_block_residency = false,
      warps_per_block_max = 1 : i32,
      semaphores_per_block = 0 : i32,
      shared_vpm_bytes = 0 : i32
    }
  } {
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %seed = ssavc4.load_imm <splat32> {value = 42 : i32} : vector<16xi32>
    // CHECK: requires vc4.resource schedule_mode = cooperative_block
    ssavc4.vpm.write %row0, %seed {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal"} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
