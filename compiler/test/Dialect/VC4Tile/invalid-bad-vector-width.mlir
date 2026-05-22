// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_width attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32
} {
  // CHECK: result type must be vector<16xi32>
  %lane = vc4tile.lane_range : vector<8xi32>
  vc4tile.return
}
