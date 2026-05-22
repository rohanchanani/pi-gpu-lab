// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @barrier
vc4tile.kernel @barrier attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  public_name = "barrier",
  warps_per_block_max = 2 : i32,
  uses_shared_vpm = false,
  uses_barrier = true,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  // CHECK: vc4tile.barrier
  vc4tile.barrier {scope = #vc4tile.barrier_scope<block>}
  vc4tile.return
}
