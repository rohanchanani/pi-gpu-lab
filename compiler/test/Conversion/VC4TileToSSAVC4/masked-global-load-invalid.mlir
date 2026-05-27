// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck %s

// CHECK: error
// CHECK: requires offsets to be vc4tile.lane_range or lane_range scaled by a positive constant for the M5 affine TMU subset
vc4tile.kernel @masked_global_load_scatter attributes {
  public_name = "masked_global_load_scatter",
  arg_attrs = [{abi_name = "base", kind = "buffer", direction = "in", elem_type = "u32", type = "u32"}],
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  uses_shared_vpm = false,
  uses_barrier = false,
  semaphores_per_block = 0 : i32,
  vpm_rows_per_block = 0 : i32,
  vpm_bytes_per_block = 0 : i32
} {
^entry(%base: i32):
  %lanes = vc4tile.lane_range : vector<16xi32>
  %offsets = arith.addi %lanes, %lanes : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  %loaded = vc4tile.masked_load_global %base, %offsets, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.return
}
