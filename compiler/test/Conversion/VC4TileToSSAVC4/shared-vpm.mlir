// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @shared_vpm_vc4tile
// CHECK-DAG: #vc4.builtin_kind<vpm_base_row>
// CHECK-DAG: schedule_mode = "cooperative_block"
// CHECK-DAG: shared_vpm_bytes = 1024 : i32
// CHECK-DAG: uses_shared_vpm = true
// CHECK: ssavc4.vpm.write
// CHECK-SAME: elem_bytes = 4 : i32
// CHECK-SAME: lanes = 16 : i32
// CHECK-SAME: orientation = "horizontal"
// CHECK: ssavc4.vpm.read
// CHECK-SAME: elem_bytes = 4 : i32
// CHECK-SAME: lanes = 16 : i32
// CHECK-SAME: orientation = "vertical"
// CHECK: ssavc4.thread_end
vc4tile.kernel @shared_vpm_vc4tile attributes {
  public_name = "shared_vpm_vc4tile",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  require_full_block_residency = true,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 16 : i32,
  vpm_bytes_per_block = 1024 : i32
} {
  %row = arith.constant 0 : i32
  %values = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  %tile = vc4tile.shared_alloc {
    rows = 16 : i32,
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile
  vc4tile.shared_store %tile, %row, %values, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<row_major>
  } : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  %loaded = vc4tile.shared_load %tile, %row, %mask {
    elem_bytes = 4 : i32,
    memory_space = #vc4tile.memory_space<shared_vpm>,
    layout = #vc4tile.vpm_layout<column_major>
  } : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  vc4tile.return
}
