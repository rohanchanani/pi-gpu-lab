// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @shared_vpm
vc4tile.kernel @shared_vpm attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  public_name = "shared_vpm",
  warps_per_block_max = 4 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  require_full_block_residency = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 8 : i32,
  vpm_bytes_per_block = 512 : i32
} {
  %row = vc4tile.warp_id : i32
  %values = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  // CHECK: vc4tile.shared_alloc
  %tile = vc4tile.shared_alloc {rows = 8 : i32, elem_bytes = 4 : i32, memory_space = #vc4tile.memory_space<shared_vpm>, layout = #vc4tile.vpm_layout<row_major>} : !vc4tile.shared_tile
  // CHECK: vc4tile.shared_store
  vc4tile.shared_store %tile, %row, %values, %mask {elem_bytes = 4 : i32, memory_space = #vc4tile.memory_space<shared_vpm>, layout = #vc4tile.vpm_layout<row_major>} : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  // CHECK: vc4tile.shared_load
  %out = vc4tile.shared_load %tile, %row, %mask {elem_bytes = 4 : i32, memory_space = #vc4tile.memory_space<shared_vpm>, layout = #vc4tile.vpm_layout<row_major>} : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  vc4tile.return
}
