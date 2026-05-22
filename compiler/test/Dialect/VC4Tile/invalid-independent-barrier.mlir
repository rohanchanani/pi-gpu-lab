// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_barrier attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_barrier = true,
  semaphores_per_block = 4 : i32,
  require_full_block_residency = true
} {
  // CHECK: error:
  vc4tile.barrier
  vc4tile.return
}
