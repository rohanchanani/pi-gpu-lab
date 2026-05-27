// RUN: not vc4-opt %s -o - 2>&1 | FileCheck %s --check-prefix=INVALID

// INVALID: error
// INVALID: tile reductions currently support only kind = #vc4tile.reduce_kind<add> in M5
vc4tile.kernel @tile_reductions_invalid attributes {
  public_name = "tile_reductions_invalid",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %values = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  %bad = "vc4tile.tile_reduce"(%values, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, kind = #vc4tile.reduce_kind<max>, axis = 1 : i32} : (vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  vc4tile.return
}
