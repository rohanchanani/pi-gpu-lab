// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @control_flow_loop_tail_block_args
// CHECK: ssavc4.br
// CHECK: ssavc4.cond_br
// CHECK: ssavc4.vdw.store
// CHECK: ssavc4.thread_end
vc4tile.kernel @control_flow_loop_tail_block_args attributes {
  public_name = "control_flow_loop_tail_block_args",
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%base: i32):
  %lanes = vc4tile.lane_range : vector<16xi32>
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %limit = arith.constant 4 : i32
  cf.br ^loop(%zero, %base : i32, i32)
^loop(%i: i32, %carried_base: i32):
  %next = arith.addi %i, %one : i32
  %keep_going = arith.cmpi ult, %next, %one : i32
  cf.cond_br %keep_going,
      ^loop(%next, %carried_base : i32, i32),
      ^merge(%carried_base, %zero, %limit : i32, i32, i32)
^merge(%mbase: i32, %mlo: i32, %mlimit: i32):
  %mask = vc4tile.core_tail_mask %mlo, %mlimit : i32, i32 -> vector<16xi1>
  vc4tile.masked_store_global %mbase, %lanes, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
