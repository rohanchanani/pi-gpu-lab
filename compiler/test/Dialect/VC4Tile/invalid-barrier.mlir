// RUN: not vc4-opt %s 2>&1 | FileCheck %s

// CHECK: uses_barrier
vc4tile.kernel @bad_barrier_contract attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 2 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  require_full_block_residency = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  vc4tile.barrier {scope = #vc4tile.barrier_scope<block>}
  vc4tile.return
}
