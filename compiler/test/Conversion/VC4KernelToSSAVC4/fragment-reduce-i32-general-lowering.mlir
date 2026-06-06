// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_reduce_i32_general
// CHECK: ssavc4.cond_select
// CHECK: ssavc4.rotate {{.*}} {amount = 8 : i32}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<add>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<min>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<max>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<and>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<or>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK-NOT: vc4.qpu.
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_reduce_i32_general(%out : i32, %n : i32) attributes {
    public_name = "fragment_reduce_i32_general",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "i32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %full = vc4kernel.pred.full : !vc4kernel.pred<16>
    %c0 = arith.constant 0 : i32
    %pred = vc4kernel.pred.tail %c0, %n : i32, i32 -> !vc4kernel.pred<16>
    %lanes = vc4kernel.lane_range : vector<16xi32>
    %offsets = vc4kernel.fragment_const {value = dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi32>} : vector<16xi32>
    %add = vc4kernel.fragment_reduce %lanes, %pred {kind = #vc4kernel.reduce<add>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %min_s = vc4kernel.fragment_reduce %lanes, %pred {kind = #vc4kernel.reduce<min_s>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %max_s = vc4kernel.fragment_reduce %lanes, %pred {kind = #vc4kernel.reduce<max_s>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %and = vc4kernel.fragment_reduce %lanes, %pred {kind = #vc4kernel.reduce<bit_and>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %or = vc4kernel.fragment_reduce %lanes, %pred {kind = #vc4kernel.reduce<bit_or>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %xor = vc4kernel.fragment_reduce %lanes, %pred {kind = #vc4kernel.reduce<bit_xor>} : vector<16xi32>, !vc4kernel.pred<16> -> vector<16xi32>
    %a = vc4kernel.fragment_alu.add %add, %min_s {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %b = vc4kernel.fragment_alu.add %max_s, %and {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %c = vc4kernel.fragment_alu.add %or, %xor {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %d = vc4kernel.fragment_alu.add %a, %b {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %result = vc4kernel.fragment_alu.add %d, %c {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.vdw_store_fragment %out, %offsets, %result, %full : i32, vector<16xi32>, vector<16xi32>, !vc4kernel.pred<16>
    vc4kernel.return
  }
}
