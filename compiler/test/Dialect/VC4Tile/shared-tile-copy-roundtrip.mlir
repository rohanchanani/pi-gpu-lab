// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @shared_tile_copy_roundtrip
vc4tile.kernel @shared_tile_copy_roundtrip(%out : i32, %in : i32) attributes {
  public_name = "shared_tile_copy_roundtrip",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 16 : i32,
  vpm_bytes_per_block = 1024 : i32
} {
  %zero = arith.constant 0 : i32
  %mask = vc4tile.mask_all
  // CHECK: vc4tile.shared_tile_alloc
  %shared = "vc4tile.shared_tile_alloc"() {rows = 16 : i32, elem_bytes = 4 : i32, shape = [16, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  // CHECK: vc4tile.copy_tile
  "vc4tile.copy_tile"(%in, %shared, %zero) {shape = [16, 16], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 64 : i32} : (i32, !vc4tile.shared_tile, i32) -> ()
  // CHECK: vc4tile.transpose_view
  %view = "vc4tile.transpose_view"(%shared) {permutation = [1, 0]} : (!vc4tile.shared_tile) -> !vc4tile.shared_tile
  // CHECK: vc4tile.copy_tile
  %loaded = "vc4tile.copy_tile"(%view, %zero, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<shared_vpm>, dst_space = #vc4tile.memory_space<register>, src_layout = #vc4tile.layout<transposed_view>, dst_layout = #vc4tile.layout<row_major>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (!vc4tile.shared_tile, i32, !vc4tile.predicate) -> vector<16xi32>
  // CHECK: vc4tile.tile_store
  "vc4tile.tile_store"(%loaded, %out, %zero, %mask) {shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
