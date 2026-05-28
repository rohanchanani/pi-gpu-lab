// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh scf_tiled_transpose_loop_vc4tile generate

vc4tile.kernel @scf_tiled_transpose_loop_vc4tile(%out : i32, %in : i32) attributes {
  public_name = "scf_tiled_transpose_loop_vc4tile",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"},
    {abi_name = "in", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = true,
  uses_barrier = true,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
  vpm_rows_per_block = 16 : i32,
  vpm_bytes_per_block = 1024 : i32
} {
  %idx0 = arith.constant 0 : index
  %idx16 = arith.constant 16 : index
  %idx1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %sixteen = arith.constant 16 : i32
  %mask = vc4tile.mask_all : vector<16xi1>
  %shared = "vc4tile.shared_tile_alloc"() {rows = 16 : i32, elem_bytes = 4 : i32, shape = [16, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  scf.for %row = %idx0 to %idx16 step %idx1 {
    %row_i32 = arith.index_cast %row : index to i32
    %offset = arith.muli %row_i32, %sixteen : i32
    %tile = "vc4tile.tile_load"(%in, %offset, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
    "vc4tile.copy_tile"(%tile, %shared, %row_i32, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  }
  vc4tile.barrier {scope = #vc4tile.barrier_scope<block>}
  %view = "vc4tile.transpose_view"(%shared) {permutation = [1, 0]} : (!vc4tile.shared_tile) -> !vc4tile.shared_tile
  "vc4tile.tile_store"(%view, %out, %zero, %mask) {shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, shared_row = 0 : i32} : (!vc4tile.shared_tile, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
