// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4tile.kernel @control_flow_roundtrip
// CHECK: cf.cond_br
// CHECK: cf.br
// CHECK: vc4tile.tail_mask
vc4tile.kernel @control_flow_roundtrip attributes {public_name = "control_flow_roundtrip"} {
  %base = arith.constant 0 : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %limit = arith.constant 8 : i32
  %go = arith.cmpi ult, %zero, %one : i32
  cf.cond_br %go,
      ^then(%base, %zero, %limit : i32, i32, i32),
      ^else(%base, %zero, %limit : i32, i32, i32)
^then(%tbase: i32, %tlo: i32, %tlimit: i32):
  cf.br ^merge(%tbase, %tlo, %tlimit : i32, i32, i32)
^else(%fbase: i32, %flo: i32, %flimit: i32):
  cf.br ^merge(%fbase, %flo, %flimit : i32, i32, i32)
^merge(%mbase: i32, %mlo: i32, %mlimit: i32):
  %mask = vc4tile.tail_mask %mlo, %mlimit : i32, i32 -> vector<16xi1>
  %bias = arith.constant 5 : i32
  %bias_vec = vector.broadcast %bias : i32 to vector<16xi32>
  %value = arith.addi %lanes, %bias_vec : vector<16xi32>
  vc4tile.masked_store_global %mbase, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
