// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_identity_uniform_index attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 2 : i32
} {
  // CHECK: must not carry uniform_index
  %block = vc4tile.block_id {uniform_index = 0 : i32} : i32
  vc4tile.return
}
