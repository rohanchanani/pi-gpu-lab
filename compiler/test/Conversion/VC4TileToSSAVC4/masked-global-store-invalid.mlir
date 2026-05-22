// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck %s

vc4tile.kernel @masked_global_store_scatter attributes {
  public_name = "masked_global_store_scatter"
} {
  %base = vc4tile.program_id : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %offsets = arith.addi %lanes, %lanes : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  // CHECK: requires offsets to be the direct vc4tile.lane_range value for the M4 coalesced VDW subset
  vc4tile.masked_store_global %base, %offsets, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<affine_contiguous>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
