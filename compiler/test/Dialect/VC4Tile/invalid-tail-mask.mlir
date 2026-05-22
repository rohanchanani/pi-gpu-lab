// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_tail_mask attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32
} {
  %base = vc4tile.program_id : i32
  %limit = vc4tile.lane_range : vector<16xi32>
  // CHECK: limit type must be i32 or index
  %tail = vc4tile.tail_mask %base, %limit : i32, vector<16xi32> -> vector<16xi1>
  vc4tile.return
}
