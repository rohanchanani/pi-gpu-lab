// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @scalar_cmpi_all_predicates_lowering
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<zs>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<zc>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cs>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cc>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cs>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cc>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cs>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cc>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cs>}
// CHECK: ssavc4.make_flags
// CHECK: ssavc4.cond_select {{.*}} {cond = #vc4.cond<cc>}
// CHECK-NOT: vc4.qpu.
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @scalar_cmpi_all_predicates_lowering(%x : i32, %y : i32) attributes {
    public_name = "scalar_cmpi_all_predicates_lowering",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "y", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %eq = arith.cmpi eq, %x, %y : i32
    %ne = arith.cmpi ne, %x, %y : i32
    %slt = arith.cmpi slt, %x, %y : i32
    %sle = arith.cmpi sle, %x, %y : i32
    %sgt = arith.cmpi sgt, %x, %y : i32
    %sge = arith.cmpi sge, %x, %y : i32
    %ult = arith.cmpi ult, %x, %y : i32
    %ule = arith.cmpi ule, %x, %y : i32
    %ugt = arith.cmpi ugt, %x, %y : i32
    %uge = arith.cmpi uge, %x, %y : i32
    %r0 = arith.select %eq, %c1, %c0 : i32
    %r1 = arith.select %ne, %r0, %c0 : i32
    %r2 = arith.select %slt, %r1, %c0 : i32
    %r3 = arith.select %sle, %r2, %c0 : i32
    %r4 = arith.select %sgt, %r3, %c0 : i32
    %r5 = arith.select %sge, %r4, %c0 : i32
    %r6 = arith.select %ult, %r5, %c0 : i32
    %r7 = arith.select %ule, %r6, %c0 : i32
    %r8 = arith.select %ugt, %r7, %c0 : i32
    %r9 = arith.select %uge, %r8, %c0 : i32
    vc4kernel.return
  }
}
