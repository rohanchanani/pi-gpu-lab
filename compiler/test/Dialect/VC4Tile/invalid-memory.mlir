// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_memory attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32
} {
  %base = vc4tile.program_id : i32
  %offsets = vc4tile.lane_range : vector<16xi32>
  %bad_mask = vc4tile.lane_range : vector<16xi32>
  // CHECK: mask type must be vector<16xi1>
  %loaded = vc4tile.masked_load_global %base, %offsets, %bad_mask {elem_bytes = 4 : i32, offset_unit = #vc4tile.offset_unit<byte>, access = #vc4tile.memory_access<coalesced>} : i32, vector<16xi32>, vector<16xi32> -> vector<16xi32>
  vc4tile.return
}
