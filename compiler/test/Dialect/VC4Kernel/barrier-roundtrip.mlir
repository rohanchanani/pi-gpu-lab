// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  vc4kernel.kernel @barrier attributes {
    public_name = "barrier",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [],
    resource = {
      uses_vpm = false,
      uses_barrier = true,
      require_full_block_residency = true,
      warps_per_block_max = 1 : i32,
      vpm_rows_per_block = 0 : i32,
      vpm_bytes_per_block = 0 : i32,
      semaphores_per_block = 4 : i32
    }
  } {
    // CHECK: schedule_mode = #vc4kernel.schedule_mode<cooperative_block>
    // CHECK: vc4kernel.barrier
    vc4kernel.barrier
    vc4kernel.return
  }
}
