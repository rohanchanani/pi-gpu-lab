// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh tile_rect_mask_store_vc4tile generate

vc4tile.kernel @tile_rect_mask_store_vc4tile(%out : i32) attributes {
  public_name = "tile_rect_mask_store_vc4tile",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", elem_type = "u32", type = "u32"}
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
  %off1 = arith.constant 16 : i32
  %off2 = arith.constant 32 : i32
  %off3 = arith.constant 48 : i32
  %off4 = arith.constant 64 : i32
  %off5 = arith.constant 80 : i32
  %values = vc4tile.lane_range : vector<16xi32>
  %mask_1x1 = vc4tile.tile_rect_mask {active_rows = 1 : i32, active_cols = 1 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>} : vector<16xi1>
  %mask_1x4 = vc4tile.tile_rect_mask {active_rows = 1 : i32, active_cols = 4 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>} : vector<16xi1>
  %mask_2x3 = vc4tile.tile_rect_mask {active_rows = 2 : i32, active_cols = 3 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>} : vector<16xi1>
  %mask_3x2 = vc4tile.tile_rect_mask {active_rows = 3 : i32, active_cols = 2 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>} : vector<16xi1>
  %mask_4x1 = vc4tile.tile_rect_mask {active_rows = 4 : i32, active_cols = 1 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>} : vector<16xi1>
  %mask_4x4 = vc4tile.tile_rect_mask {active_rows = 4 : i32, active_cols = 4 : i32, shape = [4, 4], layout = #vc4tile.layout<row_major>} : vector<16xi1>
  "vc4tile.tile_store"(%values, %out, %zero, %mask_1x1) {
    shape = [4, 4], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    boundary = #vc4tile.boundary_policy<exact>, role = #vc4tile.role<output>
  } : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  "vc4tile.tile_store"(%values, %out, %off1, %mask_1x4) {
    shape = [4, 4], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    boundary = #vc4tile.boundary_policy<exact>, role = #vc4tile.role<output>
  } : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  "vc4tile.tile_store"(%values, %out, %off2, %mask_2x3) {
    shape = [4, 4], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    boundary = #vc4tile.boundary_policy<exact>, role = #vc4tile.role<output>
  } : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  "vc4tile.tile_store"(%values, %out, %off3, %mask_3x2) {
    shape = [4, 4], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    boundary = #vc4tile.boundary_policy<exact>, role = #vc4tile.role<output>
  } : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  "vc4tile.tile_store"(%values, %out, %off4, %mask_4x1) {
    shape = [4, 4], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    boundary = #vc4tile.boundary_policy<exact>, role = #vc4tile.role<output>
  } : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  "vc4tile.tile_store"(%values, %out, %off5, %mask_4x4) {
    shape = [4, 4], dst_layout = #vc4tile.layout<row_major>, memory_space = #vc4tile.memory_space<global>,
    element_type = i32, storage_type = i32, precision = #vc4tile.precision<exact_32>, packing = #vc4tile.packing<none>,
    boundary = #vc4tile.boundary_policy<exact>, role = #vc4tile.role<output>
  } : (vector<16xi32>, i32, i32, vector<16xi1>) -> ()
  vc4tile.return
}
