// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_resource_intent attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  resource_intent = {uses_shared_vpm = true, warps_per_block = 1 : i32}
} {
  // CHECK: resource_intent.uses_shared_vpm must match uses_shared_vpm
  vc4tile.return
}
