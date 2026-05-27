// RUN: not vc4-opt %s --split-input-file 2>&1 | FileCheck %s

// CHECK: rows must be in range [1, 64]
vc4tile.kernel @reject_overlarge_shared_tile attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 16 : i32,
  vpm_bytes_per_block = 1024 : i32
} {
  %shared = "vc4tile.shared_tile_alloc"() {rows = 65 : i32, elem_bytes = 4 : i32, shape = [65, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  vc4tile.return
}

// -----

// CHECK: transpose_view supports only rank-2 permutations in M5
vc4tile.kernel @reject_transpose_rank attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 1 : i32,
  vpm_bytes_per_block = 64 : i32
} {
  %shared = "vc4tile.shared_tile_alloc"() {rows = 1 : i32, elem_bytes = 4 : i32, shape = [1, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %bad = "vc4tile.transpose_view"(%shared) {permutation = [2, 1, 0]} : (!vc4tile.shared_tile) -> !vc4tile.shared_tile
  vc4tile.return
}
