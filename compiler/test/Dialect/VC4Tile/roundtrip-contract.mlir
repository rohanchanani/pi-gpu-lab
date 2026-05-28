// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @independent
vc4tile.kernel @independent attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  public_name = "independent",
  arg_attrs = [
    {abi_name = "out", direction = "out", elem_type = "u32", kind = "buffer", type = "u32"},
    {abi_name = "in", direction = "in", elem_type = "u32", kind = "buffer", type = "u32"},
    {abi_name = "n", direction = "by_value", kind = "scalar", type = "u32"}
  ],
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
^entry(%out : i32, %in : i32, %n : i32):
  %zero = arith.constant 0 : i32
  // CHECK: vc4tile.program_id
  %pid = vc4tile.program_id : i32
  // CHECK: vc4tile.lane_id
  %lane_scalar = vc4tile.lane_id : i32
  // CHECK: vc4tile.lane_range
  %lanes = vc4tile.lane_range : vector<16xi32>
  // CHECK: vc4tile.thread_id
  %tid = vc4tile.thread_id : vector<16xi32>
  // CHECK: vc4tile.core_mask_all
  %all = vc4tile.core_mask_all : vector<16xi1>
  // CHECK: vc4tile.core_tail_mask
  %tail = vc4tile.core_tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  // CHECK: vc4tile.rotate
  %rot = vc4tile.rotate %lanes {amount = 1 : i32} : vector<16xi32> -> vector<16xi32>
  // CHECK: vc4tile.reduce
  %red = vc4tile.reduce %rot, %all {kind = #vc4tile.reduce_kind<add>} : vector<16xi32>, vector<16xi1> -> vector<16xi32>
  // CHECK: vc4tile.masked_load_global
  %ld = vc4tile.masked_load_global %in, %lanes, %tail {elem_bytes = 4 : i32, offset_unit = #vc4tile.offset_unit<byte>, memory_space = #vc4tile.memory_space<global>, access = #vc4tile.memory_access<coalesced>} : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  // CHECK: vc4tile.masked_store_global
  vc4tile.masked_store_global %out, %lanes, %red, %tail {elem_bytes = 4 : i32, offset_unit = #vc4tile.offset_unit<byte>, memory_space = #vc4tile.memory_space<global>, access = #vc4tile.memory_access<affine_contiguous>} : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  // CHECK: vc4tile.return
  vc4tile.return
}

// CHECK-LABEL: vc4tile.kernel @cooperative
vc4tile.kernel @cooperative attributes {
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  public_name = "cooperative",
  warps_per_block_max = 4 : i32,
  uses_shared_vpm = true,
  uses_barrier = true,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32,
  vpm_rows_per_block = 8 : i32,
  vpm_bytes_per_block = 512 : i32
} {
  // CHECK: vc4tile.block_id
  %block = vc4tile.block_id : i32
  // CHECK: vc4tile.warp_id
  %warp = vc4tile.warp_id : i32
  %rows = vc4tile.shared_alloc {rows = 4 : i32, elem_bytes = 4 : i32, memory_space = #vc4tile.memory_space<shared_vpm>, layout = #vc4tile.vpm_layout<row_major>} : !vc4tile.shared_tile
  %lanes = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  // CHECK: vc4tile.shared_store
  vc4tile.shared_store %rows, %warp, %lanes, %mask {elem_bytes = 4 : i32, memory_space = #vc4tile.memory_space<shared_vpm>, layout = #vc4tile.vpm_layout<row_major>} : !vc4tile.shared_tile, i32, vector<16xi32>, vector<16xi1>
  // CHECK: vc4tile.barrier
  vc4tile.barrier {scope = #vc4tile.barrier_scope<block>}
  // CHECK: vc4tile.shared_load
  %shared = vc4tile.shared_load %rows, %warp, %mask {elem_bytes = 4 : i32, memory_space = #vc4tile.memory_space<shared_vpm>, layout = #vc4tile.vpm_layout<row_major>} : !vc4tile.shared_tile, i32, vector<16xi1> -> vector<16xi32>
  vc4tile.return
}
