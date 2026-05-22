// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @masked_global_store_tail
// CHECK: ssavc4.alu.add
// CHECK: ssavc4.vdw.store
// CHECK: active_lanes = 13 : i32
// CHECK: elem_bytes = 4 : i32
// CHECK: ssavc4.thread_end
vc4tile.kernel @masked_global_store_tail attributes {
  public_name = "masked_global_store_tail",
  launch_abi = {
    public_name = "masked_global_store_tail",
    code_symbol = "masked_global_store_tail_shader",
    tail_policy = "exact_multiple",
    uniform_words_per_qpu = 1 : i32,
    args = [{direction = "by_value", kind = "scalar", name = "out", type = "u32", uniform_index = 0 : i32}],
    builtins = []
  }
} {
  %base_addr = vc4tile.program_id : i32
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
