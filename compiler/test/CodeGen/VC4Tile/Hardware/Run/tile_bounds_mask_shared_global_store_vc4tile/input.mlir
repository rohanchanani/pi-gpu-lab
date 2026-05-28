// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh tile_bounds_mask_shared_global_store_vc4tile generate

vc4tile.kernel @tile_bounds_mask_shared_global_store_vc4tile(%out : i32, %rows : i32, %cols : i32) attributes {
  public_name = "tile_bounds_mask_shared_global_store_vc4tile",
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
  %one = arith.constant 1 : i32
  %two = arith.constant 2 : i32
  %three = arith.constant 3 : i32
  %b0 = arith.constant 1000 : i32
  %b1 = arith.constant 1100 : i32
  %b2 = arith.constant 1200 : i32
  %b3 = arith.constant 1300 : i32
  %v0b = vector.broadcast %b0 : i32 to vector<16xi32>
  %v1b = vector.broadcast %b1 : i32 to vector<16xi32>
  %v2b = vector.broadcast %b2 : i32 to vector<16xi32>
  %v3b = vector.broadcast %b3 : i32 to vector<16xi32>
  %lanes = vc4tile.lane_range : vector<16xi32>
  %v0 = arith.addi %lanes, %v0b : vector<16xi32>
  %v1 = arith.addi %lanes, %v1b : vector<16xi32>
  %v2 = arith.addi %lanes, %v2b : vector<16xi32>
  %v3 = arith.addi %lanes, %v3b : vector<16xi32>
  %all = vc4tile.mask_all : vector<16xi1>
  %mask = vc4tile.tile_bounds_mask %rows, %cols {shape = [4, 4], layout = #vc4tile.layout<row_major>} : i32, i32 -> vector<16xi1>
  %shared = "vc4tile.shared_tile_alloc"() {
    rows = 4 : i32, elem_bytes = 4 : i32, shape = [4, 16],
    element_type = i32, storage_type = i32, layout = #vc4tile.layout<vpm_row>,
    role = #vc4tile.role<scratch>, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, memory_space = #vc4tile.memory_space<shared_vpm>
  } : () -> !vc4tile.shared_tile
  "vc4tile.copy_tile"(%v0, %shared, %zero, %all) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  "vc4tile.copy_tile"(%v1, %shared, %one, %all) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  "vc4tile.copy_tile"(%v2, %shared, %two, %all) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  "vc4tile.copy_tile"(%v3, %shared, %three, %all) {shape = [1, 16], src_space = #vc4tile.memory_space<register>, dst_space = #vc4tile.memory_space<shared_vpm>, src_layout = #vc4tile.layout<row_major>, dst_layout = #vc4tile.layout<vpm_row>, element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>, elem_bytes = 4 : i32} : (vector<16xi32>, !vc4tile.shared_tile, i32, vector<16xi1>) -> ()
  "vc4tile.copy_tile"(%shared, %zero, %out, %zero, %mask) {
    shape = [4, 4], src_space = #vc4tile.memory_space<shared_vpm>, dst_space = #vc4tile.memory_space<global>,
    src_layout = #vc4tile.layout<vpm_row>, dst_layout = #vc4tile.layout<row_major>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>,
    packing = #vc4tile.packing<none>, elem_bytes = 4 : i32, memory_pitch_bytes = 16 : i32
  } : (!vc4tile.shared_tile, i32, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
