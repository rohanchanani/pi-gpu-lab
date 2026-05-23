// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @masked_global_store_tail
// CHECK: ssavc4.alu.add
// CHECK: ssavc4.vdw.store %{{.*}}, %{{.*}}, %{{.*}} {active_lanes = 16 : i32
// CHECK: elem_bytes = 4 : i32
// CHECK: ssavc4.thread_end
vc4tile.kernel @masked_global_store_tail attributes {
  public_name = "masked_global_store_tail",
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%base_addr: i32):
  %lanes = vc4tile.lane_range : vector<16xi32>
  %zero = arith.constant 0 : i32
  %limit = arith.constant 13 : i32
  %mask = vc4tile.tail_mask %zero, %limit : i32, i32 -> vector<16xi1>
  vc4tile.masked_store_global %base_addr, %lanes, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
