// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
module {
  // CHECK-LABEL: vc4kernel.kernel @independent_resources
  // CHECK-SAME: require_full_block_residency = false
  // CHECK-SAME: semaphores_per_block = 0
  // CHECK-SAME: uses_barrier = false
  // CHECK-SAME: uses_vpm = false
  // CHECK-SAME: vpm_bytes_per_block = 0
  // CHECK-SAME: vpm_rows_per_block = 0
  // CHECK-SAME: warps_per_block_max = 1
  // CHECK-SAME: schedule_mode = #vc4kernel.schedule_mode<independent_vector>
  vc4kernel.kernel @independent_resources attributes {
    public_name = "independent_resources",
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
    vc4kernel.return
  }

  // CHECK-LABEL: vc4kernel.kernel @cooperative_resources
  // CHECK-SAME: require_full_block_residency = true
  // CHECK-SAME: semaphores_per_block = 4
  // CHECK-SAME: uses_barrier = true
  // CHECK-SAME: uses_vpm = true
  // CHECK-SAME: vpm_bytes_per_block = 64
  // CHECK-SAME: vpm_rows_per_block = 1
  // CHECK-SAME: warps_per_block_max = 1
  // CHECK-SAME: schedule_mode = #vc4kernel.schedule_mode<cooperative_block>
  vc4kernel.kernel @cooperative_resources attributes {
    public_name = "cooperative_resources",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [],
    resource = {
      uses_vpm = true,
      uses_barrier = true,
      require_full_block_residency = true,
      warps_per_block_max = 1 : i32,
      vpm_rows_per_block = 1 : i32,
      vpm_bytes_per_block = 64 : i32,
      semaphores_per_block = 4 : i32
    }
  } {
    vc4kernel.return
  }
}
