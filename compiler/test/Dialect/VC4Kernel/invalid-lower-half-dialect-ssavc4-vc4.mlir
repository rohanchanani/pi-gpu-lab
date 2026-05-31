// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @ssavc4_bad attributes {
    public_name = "ssavc4_bad",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    resource = {
      uses_vpm = false,
      uses_barrier = false,
      require_full_block_residency = false,
      warps_per_block_max = 1 : i32,
      vpm_rows_per_block = 0 : i32,
      vpm_bytes_per_block = 0 : i32,
      semaphores_per_block = 0 : i32
    }
  } {
    // CHECK: lower-half dialect operations are forbidden
    "ssavc4.thread_end"() : () -> ()
  }
}
