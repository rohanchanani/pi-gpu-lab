// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @f32_vector_compute(%x: f32 {vc4value.arg_name = "x"},
                              %y: f32 {vc4value.arg_name = "y"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite"} {
  %xv = vector.broadcast %x : f32 to vector<16xf32>
  %yv = vector.broadcast %y : f32 to vector<16xf32>
  %add = arith.addf %xv, %yv : vector<16xf32>
  %sub = arith.subf %add, %xv : vector<16xf32>
  %mul = arith.mulf %sub, %yv : vector<16xf32>
  %olt = arith.cmpf olt, %mul, %xv : vector<16xf32>
  %oge = arith.cmpf oge, %mul, %yv : vector<16xf32>
  %sel0 = arith.select %olt, %mul, %xv : vector<16xi1>, vector<16xf32>
  %sel1 = arith.select %oge, %sel0, %yv : vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL: vc4kernel.kernel @f32_vector_compute
// CHECK: vc4kernel.fragment_alu.add
// CHECK-SAME: opcode = #vc4kernel.add_alu_opcode<fadd>
// CHECK: vc4kernel.fragment_alu.add
// CHECK-SAME: opcode = #vc4kernel.add_alu_opcode<fsub>
// CHECK: vc4kernel.fragment_alu.mul
// CHECK-SAME: opcode = #vc4kernel.mul_alu_opcode<fmul>
// CHECK: vc4kernel.fragment_cmp
// CHECK-SAME: fp_policy = #vc4kernel.fp_cmp_policy<finite_only>
// CHECK-SAME: predicate = #vc4kernel.cmp<olt>
// CHECK: vc4kernel.fragment_cmp
// CHECK-SAME: fp_policy = #vc4kernel.fp_cmp_policy<finite_only>
// CHECK-SAME: predicate = #vc4kernel.cmp<oge>
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
