// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @masked_global_store
// CHECK: ssavc4.uniform.read
// CHECK: ssavc4.element_number
// CHECK: ssavc4.alu.add
// CHECK-NOT: ssavc4.make_flags
// CHECK: ssavc4.vdw.store
// CHECK: active_lanes = 16 : i32
// CHECK: elem_bytes = 4 : i32
// CHECK: serialize = "mutex"
// CHECK: vpm_row = 0 : i32
// CHECK: ssavc4.thread_end
// CHECK-NOT: vc4tile.
vc4tile.kernel @masked_global_store attributes {
  public_name = "masked_global_store",
  arg_attrs = [{name = "out", kind = "scalar", direction = "by_value", type = "u32"}]
} {
^entry(%base: i32):
  %lanes = vc4tile.lane_range : vector<16xi32>
  %bias = arith.constant 100 : i32
  %bias_vec = vector.broadcast %bias : i32 to vector<16xi32>
  %value = arith.addi %lanes, %bias_vec : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  vc4tile.masked_store_global %base, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
