// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh shared_transpose_store_global_vc4tile generate

vc4tile.kernel @shared_transpose_store_global_vc4tile(%out : i32, %in : i32) attributes {
  public_name = "shared_transpose_store_global_vc4tile",
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
  %mask = vc4tile.mask_all : vector<16xi1>
  %row0 = arith.constant 0 : i32
  %off0 = arith.constant 0 : i32
  %row1 = arith.constant 1 : i32
  %off1 = arith.constant 16 : i32
  %row2 = arith.constant 2 : i32
  %off2 = arith.constant 32 : i32
  %row3 = arith.constant 3 : i32
  %off3 = arith.constant 48 : i32
  %row4 = arith.constant 4 : i32
  %off4 = arith.constant 64 : i32
  %row5 = arith.constant 5 : i32
  %off5 = arith.constant 80 : i32
  %row6 = arith.constant 6 : i32
  %off6 = arith.constant 96 : i32
  %row7 = arith.constant 7 : i32
  %off7 = arith.constant 112 : i32
  %row8 = arith.constant 8 : i32
  %off8 = arith.constant 128 : i32
  %row9 = arith.constant 9 : i32
  %off9 = arith.constant 144 : i32
  %row10 = arith.constant 10 : i32
  %off10 = arith.constant 160 : i32
  %row11 = arith.constant 11 : i32
  %off11 = arith.constant 176 : i32
  %row12 = arith.constant 12 : i32
  %off12 = arith.constant 192 : i32
  %row13 = arith.constant 13 : i32
  %off13 = arith.constant 208 : i32
  %row14 = arith.constant 14 : i32
  %off14 = arith.constant 224 : i32
  %row15 = arith.constant 15 : i32
  %off15 = arith.constant 240 : i32
  %shared = "vc4tile.shared_tile_alloc"() {rows = 16 : i32, elem_bytes = 4 : i32, shape = [16, 16], element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>, role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>} : () -> !vc4tile.shared_tile
  %ld0 = "vc4tile.tile_load"(%in, %off0, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld0, %shared, %row0, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %ld1 = "vc4tile.tile_load"(%in, %off1, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld1, %shared, %row1, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %ld2 = "vc4tile.tile_load"(%in, %off2, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld2, %shared, %row2, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %ld3 = "vc4tile.tile_load"(%in, %off3, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld3, %shared, %row3, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %ld4 = "vc4tile.tile_load"(%in, %off4, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld4, %shared, %row4, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %ld5 = "vc4tile.tile_load"(%in, %off5, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld5, %shared, %row5, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %ld6 = "vc4tile.tile_load"(%in, %off6, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld6, %shared, %row6, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %ld7 = "vc4tile.tile_load"(%in, %off7, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld7, %shared, %row7, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %ld8 = "vc4tile.tile_load"(%in, %off8, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld8, %shared, %row8, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %ld9 = "vc4tile.tile_load"(%in, %off9, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld9, %shared, %row9, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %ld10 = "vc4tile.tile_load"(%in, %off10, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld10, %shared, %row10, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %ld11 = "vc4tile.tile_load"(%in, %off11, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld11, %shared, %row11, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %ld12 = "vc4tile.tile_load"(%in, %off12, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld12, %shared, %row12, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %ld13 = "vc4tile.tile_load"(%in, %off13, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld13, %shared, %row13, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %ld14 = "vc4tile.tile_load"(%in, %off14, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld14, %shared, %row14, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  %ld15 = "vc4tile.tile_load"(%in, %off15, %mask) {shape = [1, 16], src_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>} : (i32, i32, vector<16xi1>) -> vector<16xi32>
  "vc4tile.copy_tile"(%ld15, %shared, %row15, %mask) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  vc4tile.barrier {scope = #vc4tile.barrier_scope<block>}
  %view = "vc4tile.transpose_view"(%shared) {permutation = [1, 0]} : (!vc4tile.shared_tile) -> !vc4tile.shared_tile
  "vc4tile.tile_store"(%view, %out, %off0, %mask) {shape = [1, 16], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, shared_row = 0 : i32} : (!vc4tile.shared_tile, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
