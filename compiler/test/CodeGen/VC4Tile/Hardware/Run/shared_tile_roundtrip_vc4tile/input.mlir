// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh shared_tile_roundtrip_vc4tile generate

vc4tile.kernel @shared_tile_roundtrip_vc4tile(%out : i32) attributes {
  public_name = "shared_tile_roundtrip_vc4tile",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  require_full_block_residency = true,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 1 : i32,
  vpm_bytes_per_block = 64 : i32
} {
  %mask = vc4tile.mask_all : vector<16xi1>
  %row0 = arith.constant 0 : i32
  %off0 = arith.constant 0 : i32
  %shared = "vc4tile.shared_tile_alloc"() {rows = 1 : i32, elem_bytes = 4 : i32, shape = [1, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %lanes = vc4tile.lane_range : vector<16xi32>
  "vc4tile.copy_tile"(%lanes, %shared, %row0, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %loaded = "vc4tile.copy_tile"(%shared, %row0, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<shared_vpm>, dst_space = #vc4tile.memory_space<register>, src_layout = #vc4tile.layout<vpm_row>, dst_layout = #vc4tile.layout<row_major>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (!vc4tile.shared_tile, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.tile_store"(%loaded, %out, %off0, %mask) {shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
