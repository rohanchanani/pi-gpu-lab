// RUN: not vc4-opt %s 2>&1 | FileCheck %s

vc4tile.kernel @invalid_divergent_side_effecting_branch attributes {
  public_name = "invalid_divergent_side_effecting_branch"
} {
  %base = arith.constant 0 : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %value = arith.constant dense<0> : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  %one = arith.constant 1 : i32
  %pred = arith.cmpi eq, %base, %one : i32
  cf.cond_br %pred, ^store, ^exit
^store:
  // CHECK: error
  vc4tile.masked_store_global %base, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<generic>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  cf.br ^exit
^exit:
  vc4tile.return
}
