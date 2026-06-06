// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck --implicit-check-not="#vc4.add_opcode<itof>" --implicit-check-not="#vc4.add_opcode<ftoi>" --implicit-check-not="vc4.qpu." --implicit-check-not="vc4kernel." %s

// CHECK-LABEL: ssavc4.func @scalar_bitcast_i1_boundary_lowering
// CHECK-DAG: ssavc4.mov {{.*}} : i32 -> f32
// CHECK-DAG: ssavc4.mov {{.*}} : f32 -> i32
// CHECK-DAG: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<and>}
// CHECK-DAG: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<or>}
// CHECK-DAG: ssavc4.alu.add {{.*}} {opcode = #vc4.add_opcode<xor>}
// CHECK-DAG: ssavc4.cond_select
module {
  vc4kernel.kernel @scalar_bitcast_i1_boundary_lowering(%x : i32, %y : i32) attributes {
    public_name = "scalar_bitcast_i1_boundary_lowering",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "x", kind = "scalar", direction = "by_value", type = "i32"},
      {name = "y", kind = "scalar", direction = "by_value", type = "i32"}
    ],
    warps_per_block = 1 : i32
  } {
    %as_f = arith.bitcast %x : i32 to f32
    %as_i = arith.bitcast %as_f : f32 to i32
    %gt = arith.cmpi sgt, %as_i, %y : i32
    %low = arith.trunci %as_i : i32 to i1
    %and = arith.andi %gt, %low : i1
    %or = arith.ori %gt, %low : i1
    %xor = arith.xori %gt, %low : i1
    %and_i = arith.extui %and : i1 to i32
    %or_i = arith.extui %or : i1 to i32
    %xor_i = arith.extui %xor : i1 to i32
    %sum0 = arith.addi %and_i, %or_i : i32
    %sum1 = arith.addi %sum0, %xor_i : i32
    %selected = arith.select %xor, %sum1, %as_i : i32
    vc4kernel.return
  }
}
