// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @reject_sub32_surface(%in : i32, %n : i32) attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %zero = arith.constant 0 : i32
  %mask = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  // CHECK: M5 supports only 32-bit executable tile element types
  %tile = "vc4tile.tile_load"(%in, %zero, %mask) {
    shape = [1, 16],
    src_layout = #vc4tile.layout<row_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = f16,
    storage_type = f16,
    precision = #vc4tile.precision<exact_32>
  } : (i32, i32, vector<16xi1>) -> vector<16xf32>
  vc4tile.return
}
