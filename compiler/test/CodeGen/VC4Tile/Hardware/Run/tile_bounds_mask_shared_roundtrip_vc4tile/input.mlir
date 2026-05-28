// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh tile_bounds_mask_shared_roundtrip_vc4tile generate

vc4tile.kernel @tile_bounds_mask_shared_roundtrip_vc4tile(%out : i32, %rows : i32, %cols : i32) attributes {
  public_name = "tile_bounds_mask_shared_roundtrip_vc4tile",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "rows", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "cols", kind = "scalar", direction = "by_value", type = "u32"}
  ],
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = false,
  require_full_block_residency = true,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 4 : i32,
  vpm_bytes_per_block = 256 : i32
} {
  %zero = arith.constant 0 : i32
  %bias = arith.constant 100 : i32
  %biasv = vector.broadcast %bias : i32 to vector<16xi32>
  %lanes = vc4tile.lane_range : vector<16xi32>
  %values = arith.addi %lanes, %biasv : vector<16xi32>
  %mask = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  %all = vc4tile.mask_all : vector<16xi1>
  %shared = "vc4tile.shared_tile_alloc"() {
    rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16],
    element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>,
    role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>
  } : () -> !vc4tile.shared_tile
  "vc4tile.copy_tile"(%values, %shared, %zero, %mask) {
    shape = [4, 4], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>,
    src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, elem_bytes = 4 : i32
  } : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %roundtrip = "vc4tile.copy_tile"(%shared, %zero, %mask) {
    shape = [4, 4], src_space = #vc4tile.memory_space<shared_vpm>, dst_space = #vc4tile.memory_space<register>,
    src_layout = #vc4tile.layout<vpm_row>, dst_layout = #vc4tile.layout<row_major>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, elem_bytes = 4 : i32
  } : (!vc4tile.shared_tile, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.tile_store"(%roundtrip, %out, %zero, %all) {
    shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>
  } : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
