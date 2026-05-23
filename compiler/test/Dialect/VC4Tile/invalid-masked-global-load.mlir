// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @invalid_masked_global_load_generic_access attributes {
  public_name = "invalid_masked_global_load_generic_access"
} {
  %base = arith.constant 0 : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  // CHECK: generic per-lane load access is outside the M4 hardware-lowered contract
  %loaded = vc4tile.masked_load_global %base, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<generic>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.return
}
