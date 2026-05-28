// RUN: not vc4-opt %s 2>&1 | FileCheck %s

// CHECK: contains shared VPM ops but uses_shared_vpm is not true
vc4tile.kernel @bad_shared_contract attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  require_full_block_residency = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %row = vc4tile.warp_id : i32
  %values = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  %tile = vc4tile.shared_alloc {
    rows = 1 : i32,
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile
  vc4tile.shared_store %tile, %row, %values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
