// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh tile_reduce_tail_sum_vc4tile generate

vc4tile.kernel @tile_reduce_tail_sum_vc4tile(%out : i32, %in : i32, %n : i32) attributes {
  public_name = "tile_reduce_tail_sum_vc4tile",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"}
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
  %all = vc4tile.mask_all : vector<16xi1>
  %tail = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %tile = "vc4tile.tile_load"(%in, %zero, %all) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %sum = "vc4tile.tile_reduce"(%tile, %tail) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, kind = #vc4tile.reduce_kind<add>, axis = 1 : i32, algorithm_hint = "predicated_rotate_add_tree"} : (vector<16xi32>, vector<16xi1>) -> vector<16xi32>
  "vc4tile.tile_store"(%sum, %out, %zero, %all) {shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<exact>, packing = #vc4tile.packing<none>, role = #vc4tile.role<output>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
