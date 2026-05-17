// RUN: not vc4-opt %s --convert-ssavc4-to-vc4 2>&1 | FileCheck %s

ssavc4.module @barrier_invalid_resource {
  ssavc4.func @bad_barrier_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      warps_per_block_max = 1 : i32,
      uses_shared_vpm = false,
      uses_barrier = false,
      semaphores_per_block = 0 : i32,
      require_full_block_residency = false
    }
  } {
    // CHECK: requires vc4.resource schedule_mode = "cooperative_block"
    ssavc4.barrier {arrive_offset = 0 : i32, go_offset = 1 : i32, depart_offset = 2 : i32, reset_offset = 3 : i32}
    ssavc4.thread_end
  }
}
