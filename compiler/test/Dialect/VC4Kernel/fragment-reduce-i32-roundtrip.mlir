// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s

module {
  vc4kernel.kernel @fragment_reduce_i32(%x : i32, %n : i32) attributes {
    public_name = "fragment_reduce_i32",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %xv = vc4kernel.splat %x : i32 -> vector<16xi32>
    %c0 = arith.constant 0 : i32
    %pred = vc4kernel.pred.tail %c0, %n : i32, i32 -> !vc4kernel.pred<16>
    // CHECK: #vc4kernel.reduce<add>
    %add = vc4kernel.fragment_reduce %xv, %pred {kind = #vc4kernel.reduce<add>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    // CHECK: #vc4kernel.reduce<min_s>
    %min_s = vc4kernel.fragment_reduce %lanes, %pred {kind = #vc4kernel.reduce<min_s>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    // CHECK: #vc4kernel.reduce<max_s>
    %max_s = vc4kernel.fragment_reduce %lanes, %pred {kind = #vc4kernel.reduce<max_s>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    // CHECK: #vc4kernel.reduce<min_u>
    %min_u = vc4kernel.fragment_reduce %lanes, %pred {kind = #vc4kernel.reduce<min_u>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    // CHECK: #vc4kernel.reduce<max_u>
    %max_u = vc4kernel.fragment_reduce %lanes, %pred {kind = #vc4kernel.reduce<max_u>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    // CHECK: #vc4kernel.reduce<bit_and>
    %and = vc4kernel.fragment_reduce %lanes, %pred {kind = #vc4kernel.reduce<bit_and>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    // CHECK: #vc4kernel.reduce<bit_or>
    %or = vc4kernel.fragment_reduce %lanes, %pred {kind = #vc4kernel.reduce<bit_or>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    // CHECK: #vc4kernel.reduce<bit_xor>
    %xor = vc4kernel.fragment_reduce %lanes, %pred {kind = #vc4kernel.reduce<bit_xor>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %use0 = vc4kernel.fragment_alu.add %add, %min_s {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %use1 = vc4kernel.fragment_alu.add %max_s, %min_u {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %use2 = vc4kernel.fragment_alu.add %max_u, %and {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %use3 = vc4kernel.fragment_alu.add %or, %xor {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %sink0 = vc4kernel.fragment_alu.add %use0, %use1 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %sink1 = vc4kernel.fragment_alu.add %use2, %use3 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %sink = vc4kernel.fragment_alu.add %sink0, %sink1 {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.return
  }
}
