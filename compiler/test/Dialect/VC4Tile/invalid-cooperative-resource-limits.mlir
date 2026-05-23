// RUN: not vc4-opt %s 2>&1 | FileCheck %s

// CHECK: vpm_bytes_per_block must fit vpm_rows_per_block
vc4tile.kernel @bad_cooperative_resource_limits attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 2 : i32,
  vpm_rows_per_block = 1 : i32,
  vpm_bytes_per_block = 128 : i32,
  semaphores_per_block = 0 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  require_full_block_residency = false
} {
  vc4tile.return
}
