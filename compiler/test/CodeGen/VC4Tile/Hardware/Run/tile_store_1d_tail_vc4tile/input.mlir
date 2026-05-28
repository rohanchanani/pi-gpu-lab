// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh tile_store_1d_tail_vc4tile generate

vc4tile.kernel @tile_store_1d_tail_vc4tile(%out : i32, %n : i32) attributes {
  public_name = "tile_store_1d_tail_vc4tile",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
  %zero = arith.constant 0 : i32
  %c97 = arith.constant 97 : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %bias = vector.broadcast %c97 : i32 to vector<16xi32>
  %values = arith.addi %lanes, %bias : vector<16xi32>
  %mask = vc4tile.tail_mask %zero, %n : i32, i32
  "vc4tile.tile_store"(%values, %out, %zero, %mask) {
    shape = [1, 16],
    dst_layout = #vc4tile.layout<row_major>,
    memory_space = #vc4tile.memory_space<global>,
    element_type = i32,
    storage_type = i32,
    precision = #vc4tile.precision<exact_32>,
    boundary = #vc4tile.boundary_policy<tail_predicated>,
    packing = #vc4tile.packing<none>
  } : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
