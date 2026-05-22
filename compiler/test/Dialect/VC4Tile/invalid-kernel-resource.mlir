// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_resource attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 13 : i32,
  semaphores_per_block = 0 : i32
} {
  // CHECK: warps_per_block_max must be in range [1, 12]
  vc4tile.return
}
