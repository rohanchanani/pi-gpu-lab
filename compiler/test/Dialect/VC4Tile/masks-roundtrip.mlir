// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @masks_roundtrip
vc4tile.kernel @masks_roundtrip attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  public_name = "masks_roundtrip",
  warps_per_block_max = 1 : i32,
  semaphores_per_block = 0 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  require_full_block_residency = false,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %base = arith.constant 0 : i32
  %limit = arith.constant 13 : i32
  // CHECK: vc4tile.mask_all
  %all = vc4tile.mask_all : vector<16xi1>
  // CHECK: vc4tile.tail_mask
  %tail = vc4tile.tail_mask %base, %limit : i32, i32 -> vector<16xi1>
  // CHECK: vc4tile.tile_rect_mask
  %rect = vc4tile.tile_rect_mask {active_rows = 3 : i32, active_cols = 2 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>} : vector<16xi1>
  vc4tile.return
}
