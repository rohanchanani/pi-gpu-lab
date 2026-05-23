// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @formal_args_lowering()
// CHECK-SAME: function_type = () -> ()
// CHECK-SAME: args = [
// CHECK-SAME: abi_name = "out"
// CHECK-SAME: name = "out"
// CHECK-SAME: uniform_index = 0 : i32
// CHECK-SAME: abi_name = "n"
// CHECK-SAME: name = "n"
// CHECK-SAME: uniform_index = 1 : i32
// CHECK-SAME: abi_name = "alpha"
// CHECK-SAME: name = "alpha"
// CHECK-SAME: type = "f32"
// CHECK-SAME: uniform_index = 2 : i32
// CHECK-SAME: builtins = []
// CHECK-SAME: uniform_words_per_qpu = 3 : i32
// CHECK: %[[OUT:.*]] = ssavc4.uniform.read 0 {abi_name = "out", name = "out"} : i32
// CHECK: %[[N:.*]] = ssavc4.uniform.read 1 {abi_name = "n", name = "n"} : i32
// CHECK: %[[ALPHA:.*]] = ssavc4.uniform.read 2 {abi_name = "alpha", name = "alpha"} : f32
// CHECK: ssavc4.splat %[[N]] : i32 -> vector<16xi32>
// CHECK: %[[ALPHA_VEC:.*]] = ssavc4.splat %[[ALPHA]] : f32 -> vector<16xf32>
// CHECK: ssavc4.vdw.store %[[OUT]], %[[ALPHA_VEC]]
vc4tile.kernel @formal_args_lowering(%out : i32, %n : i32, %alpha : f32) attributes {
  public_name = "formal_args_lowering",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", type = "u32", elem_type = "f32"},
    {abi_name = "n", kind = "scalar", direction = "by_value", type = "u32"},
    {abi_name = "alpha", kind = "scalar", direction = "by_value", type = "f32"}
  ]
} {
  %zero = arith.constant 0 : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.tail_mask %zero, %n : i32, i32 -> vector<16xi1>
  %alpha_vec = vector.broadcast %alpha : f32 to vector<16xf32>
  vc4tile.masked_store_global %out, %lanes, %alpha_vec, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xf32>, vector<16xi1>
  vc4tile.return
}

// CHECK-LABEL: ssavc4.func @formal_args_with_program_id()
// CHECK-SAME: args = [
// CHECK-SAME: abi_name = "out"
// CHECK-SAME: name = "out"
// CHECK-SAME: uniform_index = 0 : i32
// CHECK-SAME: builtins = [
// CHECK-SAME: kind = #vc4.builtin_kind<logical_request>
// CHECK-SAME: name = "logical_request"
// CHECK-SAME: uniform_index = 1 : i32
// CHECK-SAME: uniform_words_per_qpu = 2 : i32
// CHECK: %[[OUT2:.*]] = ssavc4.uniform.read 0 {abi_name = "out", name = "out"} : i32
// CHECK: %[[REQ:.*]] = ssavc4.uniform.read 1 {abi_name = "logical_request", name = "logical_request"} : i32
// CHECK: %[[LANES:.*]] = ssavc4.element_number : vector<16xi32>
// CHECK: %[[REQ_VEC:.*]] = ssavc4.splat %[[REQ]] : i32 -> vector<16xi32>
// CHECK: ssavc4.alu.add %[[LANES]], %[[REQ_VEC]]
// CHECK: ssavc4.vdw.store %[[OUT2]]
vc4tile.kernel @formal_args_with_program_id(%out : i32) attributes {
  public_name = "formal_args_with_program_id",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32,
  arg_attrs = [
    {abi_name = "out", kind = "buffer", direction = "out", type = "u32", elem_type = "u32"}
  ]
} {
  %pid = vc4tile.program_id : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %pid_vec = vector.broadcast %pid : i32 to vector<16xi32>
  %value = arith.addi %lanes, %pid_vec : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  vc4tile.masked_store_global %out, %lanes, %value, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<element>,
    memory_space = #vc4tile.memory_space<global>,
    access = #vc4tile.memory_access<coalesced>
  } : i32, vector<16xi32>, vector<16xi32>, vector<16xi1>
  vc4tile.return
}

// CHECK-LABEL: ssavc4.func @zero_formal_still_lowers()
// CHECK-SAME: args = []
// CHECK-SAME: builtins = []
// CHECK-SAME: uniform_words_per_qpu = 0 : i32
// CHECK: ssavc4.thread_end
vc4tile.kernel @zero_formal_still_lowers attributes {
  public_name = "zero_formal_still_lowers",
  schedule_mode = #vc4tile.schedule_mode<independent_vector>,
  warps_per_block_max = 1 : i32
} {
  vc4tile.return
}
