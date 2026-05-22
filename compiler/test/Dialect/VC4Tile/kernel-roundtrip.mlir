// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @independent_kernel
vc4tile.kernel @independent_kernel attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  public_name = "independent_kernel",
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  require_full_block_residency = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  // CHECK: vc4tile.return
  vc4tile.return
}

// CHECK-LABEL: vc4tile.kernel @cooperative_kernel
vc4tile.kernel @cooperative_kernel attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  public_name = "cooperative_kernel",
  warps_per_block_max = 4 : i32,
  uses_shared_vpm = false,
  uses_barrier = true,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  vc4tile.barrier {scope = #vc4tile.barrier_scope<block>}
  vc4tile.return
}
