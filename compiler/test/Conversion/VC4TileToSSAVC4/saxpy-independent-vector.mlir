// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @saxpy_independent_vector
// CHECK: ssavc4.uniform.read
// CHECK: ssavc4.element_number
// CHECK: ssavc4.alu.mul
// CHECK: ssavc4.alu.add
// CHECK: ssavc4.tmu.request
// CHECK: mode = "direct"
// CHECK: unit = "tmu0"
// CHECK: ssavc4.tmu.read
// CHECK: part = "raw32"
// CHECK: ssavc4.alu.mul
// CHECK: ssavc4.alu.add
// CHECK: ssavc4.vdw.store
// CHECK: ssavc4.thread_end
// CHECK-NOT: vc4tile.
vc4tile.kernel @saxpy_independent_vector attributes {
  public_name = "saxpy_independent_vector",
  launch_abi = {
    public_name = "saxpy_independent_vector",
    code_symbol = "saxpy_independent_vector_shader",
    tail_policy = "exact_multiple",
    uniform_words_per_qpu = 1 : i32,
    args = [{direction = "by_value", kind = "scalar", name = "buffer", type = "u32", uniform_index = 0 : i32}],
    builtins = []
  }
} {
  %base = vc4tile.program_id : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  %loaded = vc4tile.masked_load_global %base, %lanes, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  %two = arith.constant 2 : i32
  %two_vec = vector.broadcast %two : i32 to vector<16xi32>
  %scaled = arith.muli %loaded, %two_vec : vector<16xi32>
  %result = arith.addi %scaled, %lanes : vector<16xi32>
  vc4tile.masked_store_global %base, %lanes, %result, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}
