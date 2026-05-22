// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_store attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32
} {
  %base = vc4tile.program_id : i32
  %offsets = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  // CHECK: generic per-lane store access is outside the M4 hardware-lowered contract
  vc4tile.masked_store_global %base, %offsets, %offsets, %mask {elem_bytes = 4 : i32, offset_unit = #vc4tile.offset_unit<byte>, access = #vc4tile.memory_access<generic>} : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
