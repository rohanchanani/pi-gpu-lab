// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh tile_bounds_mask_load_vc4tile generate

vc4tile.kernel @tile_bounds_mask_load_vc4tile(%out : i32, %in : i32, %rows : i32, %cols : i32) attributes {
  public_name = "tile_bounds_mask_load_vc4tile",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "rows", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "cols", kind = "scalar", direction = "by_value", type = "u32"}
  ],
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %zero = arith.constant 0 : i32
  %mask = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32
  %tile = "vc4tile.tile_load"(%in, %zero, %mask) {
    shape = [4, 4], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    boundary = #vc4tile.boundary_policy<exact>, role = #vc4tile.role<input>
  } : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  %all = vc4tile.mask_all
  "vc4tile.tile_store"(%tile, %out, %zero, %all) {
    shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    boundary = #vc4tile.boundary_policy<exact>, role = #vc4tile.role<output>
  } : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
