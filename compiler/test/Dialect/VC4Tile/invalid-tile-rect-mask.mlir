// RUN: not vc4-opt %s -o - 2>&1 | FileCheck %s

// CHECK: error
// CHECK: active_cols must be in range [1, 4]
vc4tile.kernel @invalid_tile_rect_mask attributes {
  public_name = "invalid_tile_rect_mask",
  arg_attrs = [],
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %mask = vc4tile.tile_rect_mask {active_rows = 4 : i32, active_cols = 5 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>}
  vc4tile.return
}
