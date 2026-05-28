// RUN: not vc4-opt %s -o - 2>&1 | FileCheck %s

// CHECK: tile_bounds_mask supports only shape = [4, 4] in M5
vc4tile.kernel @invalid_tile_bounds_mask attributes {
  public_name = "invalid_tile_bounds_mask",
  arg_attrs = [],
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %rows = arith.constant 3 : i32
  %cols = arith.constant 2 : i32
  %mask = vc4tile.tile_bounds_mask %rows, %cols {shape = [8, 4], layout = #vc4tile.layout<row_major>} : i32, i32
  vc4tile.return
}
