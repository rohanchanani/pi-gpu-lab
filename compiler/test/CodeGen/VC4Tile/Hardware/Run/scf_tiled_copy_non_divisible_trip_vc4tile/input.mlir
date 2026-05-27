// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh scf_tiled_copy_non_divisible_trip_vc4tile generate

vc4tile.kernel @scf_tiled_copy_non_divisible_trip_vc4tile(%out : i32, %in : i32, %n : i32) attributes {
  public_name = "scf_tiled_copy_non_divisible_trip_vc4tile",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %lb = arith.constant 0 : index
  %n_idx = arith.index_cast %n : i32 to index
  %step = arith.constant 16 : index
  scf.for %i = %lb to %n_idx step %step {
    %i32 = arith.index_cast %i : index to i32
    %mask = vc4tile.tail_mask %i32, %n : i32, i32 -> vector<16xi1>
    %tile = "vc4tile.tile_load"(%in, %i32, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
    "vc4tile.tile_store"(%tile, %out, %i32, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  }
  vc4tile.return
}
