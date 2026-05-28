// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @invalid_masked_global_store_generic_access attributes {
  public_name = "invalid_masked_global_store_generic_access"
} {
  %base = arith.constant 0 : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.core_mask_all : vector<16xi1>
  // CHECK: generic per-lane store access is outside the M4 hardware-lowered contract
  vc4tile.masked_store_global %base, %lanes, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<generic>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
