// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh tile_contract_shared_rhs_vc4tile generate

vc4tile.kernel @tile_contract_shared_rhs_vc4tile(%out : i32, %lhs_base : i32, %rhs_base : i32) attributes {
  public_name = "tile_contract_shared_rhs_vc4tile",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "lhs", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"},
    {abi_name = "rhs", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}
  ],
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = true,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
  vpm_rows_per_block = 1 : i32,
  vpm_bytes_per_block = 64 : i32
} {
  %zero = arith.constant 0 : i32
  %mask = vc4tile.mask_all
  %lhs = "vc4tile.tile_load"(%lhs_base, %zero, %mask) {shape = [4, 4], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, !vc4tile.predicate) -> vector<16xi32>
  %shared = "vc4tile.shared_tile_alloc"() {rows = 1 : i32, elem_bytes = 4 : i32, shape = [1, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<input>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  "vc4tile.copy_tile"(%rhs_base, %shared, %zero, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<global>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_pitch_bytes = 64 : i32} : (i32, !vc4tile.shared_tile, i32, !vc4tile.predicate) -> ()
  vc4tile.barrier {scope = #vc4tile.barrier_scope<block>}
  %rhs = "vc4tile.copy_tile"(%shared, %zero, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<shared_vpm>, dst_space = #vc4tile.memory_space<register>, src_layout = #vc4tile.layout<vpm_row>, dst_layout = #vc4tile.layout<row_major>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (!vc4tile.shared_tile, i32, !vc4tile.predicate) -> vector<16xi32>
  %acc = "vc4tile.tile_fill"() {shape = [4, 4], layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, value = 13 : i32} : () -> vector<16xi32>
  %result = "vc4tile.tile_contract"(%lhs, %rhs, %acc, %mask) {m = 4 : i32, n = 4 : i32, k = 4 : i32, shape = [4, 4], lhs_layout = #vc4tile.layout<row_major>, rhs_layout = #vc4tile.layout<row_major>, acc_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<register>, element_type = i32, storage_type = i32, accumulator_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, contracting_dims = [[1], [0]], iterator_types = ["parallel", "parallel", "reduction"], lhs_role = #vc4tile.role<input>, rhs_role = #vc4tile.role<input>, acc_role = #vc4tile.role<accumulator>, algorithm_hint = "matmul_4x4x4"} : (vector<16xi32>, vector<16xi32>, vector<16xi32>, !vc4tile.predicate) -> vector<16xi32>
  "vc4tile.tile_store"(%result, %out, %zero, %mask) {shape = [4, 4], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (vector<16xi32>, i32, i32, !vc4tile.predicate) -> ()
  vc4tile.return
}
