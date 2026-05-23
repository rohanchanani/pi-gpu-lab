// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck %s

// CHECK: warps_per_block_max must be in range [1, 12]
vc4tile.kernel @bad_cooperative_resource attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 13 : i32,
  uses_shared_vpm = false,
  uses_barrier = false
} {
  vc4tile.return
}
