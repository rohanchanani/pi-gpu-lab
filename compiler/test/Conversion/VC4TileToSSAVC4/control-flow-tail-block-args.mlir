// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @control_flow_tail_block_args
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_br
// CHECK: ssavc4.br
// CHECK: ssavc4.element_number
// CHECK: ssavc4.alu.add
// CHECK: ssavc4.vdw.store
// CHECK: active_lanes = 16 : i32
// CHECK: ssavc4.thread_end
// CHECK-NOT: vc4tile.
vc4tile.kernel @control_flow_tail_block_args(%out : i32) attributes {
  public_name = "control_flow_tail_block_args",
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", type = "u32", elem_type = "u32"}
  ]
} {
  %lanes = vc4tile.lane_range : vector<16xi32>
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %limit = arith.constant 8 : i32
  %bias = arith.constant 700 : i32
  %bias_vec = vector.broadcast %bias : i32 to vector<16xi32>
  %value = arith.addi %lanes, %bias_vec : vector<16xi32>
  %go = arith.cmpi ult, %zero, %one : i32
  cf.cond_br %go,
      ^then(%out, %zero, %limit, %value : i32, i32, i32, vector<16xi32>),
      ^else(%out, %zero, %limit, %value : i32, i32, i32, vector<16xi32>)
^then(%tbase: i32, %tlo: i32, %tlimit: i32, %tvalue: vector<16xi32>):
  cf.br ^merge(%tbase, %tlo, %tlimit, %tvalue : i32, i32, i32, vector<16xi32>)
^else(%fbase: i32, %flo: i32, %flimit: i32, %fvalue: vector<16xi32>):
  cf.br ^merge(%fbase, %flo, %flimit, %fvalue : i32, i32, i32, vector<16xi32>)
^merge(%mbase: i32, %mlo: i32, %mlimit: i32, %mvalue: vector<16xi32>):
  %mask = vc4tile.tail_mask %mlo, %mlimit : i32, i32 -> vector<16xi1>
  vc4tile.masked_store_global %mbase, %lanes, %mvalue, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
