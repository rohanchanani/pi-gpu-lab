// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_barrier attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_barrier = true,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32
} {
  // CHECK: independent_vector kernels must not use barriers
  vc4tile.barrier {scope = #vc4tile.barrier_scope<block>}
  vc4tile.return
}
