// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @masked_memory
vc4tile.kernel @masked_memory attributes {
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  semaphores_per_block = 0 : i32
} {
  %base = vc4tile.program_id : i32
  %offsets = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  // CHECK: vc4tile.masked_load_global
  %loaded = vc4tile.masked_load_global %base, %offsets, %mask {elem_bytes = 4 : i32, offset_unit = #vc4tile.offset_unit<byte>, memory_space = #vc4tile.memory_space<global>, access = #vc4tile.memory_access<coalesced>} : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  // CHECK: vc4tile.masked_store_global
  vc4tile.masked_store_global %base, %offsets, %loaded, %mask {elem_bytes = 4 : i32, offset_unit = #vc4tile.offset_unit<element>, memory_space = #vc4tile.memory_space<global>, access = #vc4tile.memory_access<affine_contiguous>} : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
