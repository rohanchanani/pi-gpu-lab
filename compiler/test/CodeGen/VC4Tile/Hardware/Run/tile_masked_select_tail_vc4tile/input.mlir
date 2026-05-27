// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh tile_masked_select_tail_vc4tile generate

vc4tile.kernel @tile_masked_select_tail_vc4tile(%out : i32, %x : i32, %n : i32) attributes {
  public_name = "tile_masked_select_tail_vc4tile",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "x", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
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
  %mask = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %xv = "vc4tile.tile_load"(%x, %zero, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  %fallback = "vc4tile.tile_fill"() {value = 1234 : i32, shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, role = #vc4tile.role<constant>, packing = #vc4tile.packing<none>} : () -> vector<16xi32>
  %sel = "vc4tile.tile_select"(%mask, %xv, %fallback) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (vector<16xi1>, vector<16xi32>, vector<16xi32>) -> vector<16xi32>
  "vc4tile.tile_store"(%sel, %out, %zero, %mask) {shape = [1, 16], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, boundary = #vc4tile.boundary_policy<tail_predicated>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
