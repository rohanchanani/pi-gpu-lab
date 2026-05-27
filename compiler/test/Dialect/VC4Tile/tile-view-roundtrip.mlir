// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @tile_view_roundtrip
vc4tile.kernel @tile_view_roundtrip attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %values = vc4tile.lane_range : vector<16xi32>
  // CHECK: vc4tile.tile_view
  %view = "vc4tile.tile_view"(%values) {offsets = [1], sizes = [16], strides = [1], layout = #vc4tile.layout<row_major>} : (vector<16xi32>) -> vector<16xi32>
  // CHECK: vc4tile.tile_subview
  %sub = "vc4tile.tile_subview"(%view) {offsets = [1], sizes = [16], strides = [1], layout = #vc4tile.layout<row_major>} : (vector<16xi32>) -> vector<16xi32>
  // CHECK: vc4tile.transpose_view
  %t = "vc4tile.transpose_view"(%sub) {permutation = [1, 0]} : (vector<16xi32>) -> vector<16xi32>
  vc4tile.return
}
