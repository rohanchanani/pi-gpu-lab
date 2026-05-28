// RUN: bash %S/../../../Support/run_vc4tile_candidate_codegen_test.sh cooperative_id_writeback_vc4tile generate

vc4tile.kernel @cooperative_id_writeback_vc4tile attributes {
  public_name = "cooperative_id_writeback_vc4tile",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  warps_per_block_max = 12 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  require_full_block_residency = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32,
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%out: i32):
  %block = vc4tile.block_id : i32
  %warp = vc4tile.warp_id : i32
  %thread = vc4tile.thread_id : vector<16xi32>
  %lanes = vc4tile.lane_range : vector<16xi32>
  %six = arith.constant 6 : i32
  %thousand = arith.constant 1000 : i32
  %block_stride_bytes = arith.constant 768 : i32
  %block_bytes = arith.muli %block, %block_stride_bytes : i32
  %warp_bytes = arith.shli %warp, %six : i32
  %block_scaled = arith.muli %block, %thousand : i32
  %tmp_base = arith.addi %out, %block_bytes : i32
  %base = arith.addi %tmp_base, %warp_bytes : i32
  %block_vec = vector.broadcast %block_scaled : i32 to vector<16xi32>
  %value = arith.addi %block_vec, %thread : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  vc4tile.masked_store_global %base, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
