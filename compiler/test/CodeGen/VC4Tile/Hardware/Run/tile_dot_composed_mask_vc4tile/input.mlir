// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh tile_dot_composed_mask_vc4tile generate

vc4tile.kernel @tile_dot_composed_mask_vc4tile(%out : i32, %lhs_base : i32, %rhs_base : i32, %rows : i32, %cols : i32, %n : i32) attributes {
  public_name = "tile_dot_composed_mask_vc4tile",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "lhs", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "rhs", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "rows", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "cols", kind = "scalar", direction = "by_value", type = "u32"},
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
  %all = vc4tile.mask_all
  %bounds = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32
  %tail = vc4tile.tail_mask %zero, %n : i32, i32
  %mask = vc4tile.mask_and %bounds, %tail
  %lhs = "vc4tile.tile_load"(%lhs_base, %zero, %all) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  %rhs = "vc4tile.tile_load"(%rhs_base, %zero, %all) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  %result = "vc4tile.tile_dot"(%lhs, %rhs, %mask) {k = 16 : i32, shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "dot_1x16_composed_mask"} : (vector<16xi32>, vector<16xi32>, !vc4tile.predicate) -> vector<16xi32>
  "vc4tile.tile_store"(%result, %out, %zero, %all) {shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, role = #vc4tile.role<output>} : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
