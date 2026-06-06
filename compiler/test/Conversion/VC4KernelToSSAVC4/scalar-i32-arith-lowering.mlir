// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @scalar_i32_arith_lowering
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<add>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<sub>}
// Exact scalar arith.muli must expand to bounded 16x16 partial products.
// CHECK: #vc4.add_opcode<and>
// CHECK: #vc4.add_opcode<shr>
// CHECK-COUNT-3: ssavc4.alu.mul {{.*}} {opcode = #vc4.mul_opcode<mul24>}
// CHECK: #vc4.add_opcode<shl>
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<shl>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<shr>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<asr>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<and>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<or>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<min>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<max>}
// Unsigned min/max use sign-bias through xor, signed min/max, xor back.
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<min>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<max>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK-NOT: vc4.qpu.
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @scalar_i32_arith_lowering(%x : i32, %y : i32) attributes {
    public_name = "scalar_i32_arith_lowering",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "y", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c1 = arith.constant 1 : i32
    %add = arith.addi %x, %y : i32
    %sub = arith.subi %add, %y : i32
    %mul = arith.muli %sub, %x : i32
    %shl = arith.shli %mul, %c1 : i32
    %shrui = arith.shrui %shl, %c1 : i32
    %shrsi = arith.shrsi %shrui, %c1 : i32
    %and = arith.andi %shrsi, %x : i32
    %or = arith.ori %and, %y : i32
    %xor = arith.xori %or, %x : i32
    %minsi = arith.minsi %xor, %y : i32
    %maxsi = arith.maxsi %minsi, %x : i32
    %minui = arith.minui %maxsi, %y : i32
    %maxui = arith.maxui %minui, %x : i32
    vc4kernel.return
  }
}
