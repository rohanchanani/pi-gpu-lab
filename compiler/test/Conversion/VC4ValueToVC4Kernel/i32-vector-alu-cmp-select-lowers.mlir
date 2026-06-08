// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @i32_vector_compute(%x: i32 {vc4value.arg_name = "x"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.i32_mul_policy = "mul24_safe"} {
  %c1 = arith.constant dense<1> : vector<16xi32>
  %c2 = arith.constant dense<2> : vector<16xi32>
  %xv = vector.broadcast %x : i32 to vector<16xi32>
  %add = arith.addi %xv, %c1 : vector<16xi32>
  %sub = arith.subi %add, %c2 : vector<16xi32>
  %shl = arith.shli %sub, %c1 : vector<16xi32>
  %mul = arith.muli %shl, %c2 : vector<16xi32>
  %slt = arith.cmpi slt, %mul, %xv : vector<16xi32>
  %uge = arith.cmpi uge, %mul, %xv : vector<16xi32>
  %sel0 = arith.select %slt, %mul, %xv : vector<16xi1>, vector<16xi32>
  %sel1 = arith.select %uge, %sel0, %c1 : vector<16xi1>, vector<16xi32>
  return
}

// CHECK-LABEL: vc4kernel.kernel @i32_vector_compute
// CHECK: vc4kernel.fragment_alu.add
// CHECK-SAME: opcode = #vc4kernel.add_alu_opcode<add>
// CHECK: vc4kernel.fragment_alu.add
// CHECK-SAME: opcode = #vc4kernel.add_alu_opcode<sub>
// CHECK: vc4kernel.fragment_alu.add
// CHECK-SAME: opcode = #vc4kernel.add_alu_opcode<shl>
// CHECK: vc4kernel.fragment_alu.mul
// CHECK-SAME: opcode = #vc4kernel.mul_alu_opcode<mul24>
// CHECK: vc4kernel.fragment_cmp
// CHECK-SAME: predicate = #vc4kernel.cmp<slt>
// CHECK: vc4kernel.fragment_cmp
// CHECK-SAME: predicate = #vc4kernel.cmp<uge>
// CHECK: vc4kernel.fragment_select
// CHECK: vc4kernel.fragment_select
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
// CHECK-NOT: linalg.
// CHECK-NOT: gpu.
// CHECK-NOT: tt.
