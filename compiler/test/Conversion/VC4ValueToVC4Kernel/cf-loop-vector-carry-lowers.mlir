// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @cf_loop_vector_carry(%n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant dense<0.000000e+00> : vector<16xf32>
  %one = arith.constant dense<1.000000e+00> : vector<16xf32>
  cf.br ^loop(%c0, %zero : index, vector<16xf32>)

^loop(%i: index, %acc: vector<16xf32>):
  %done = arith.cmpi uge, %i, %n : index
  cf.cond_br %done, ^exit(%acc : vector<16xf32>), ^body(%i, %acc : index, vector<16xf32>)

^body(%body_i: index, %body_acc: vector<16xf32>):
  %next_acc = arith.addf %body_acc, %one : vector<16xf32>
  %next_i = arith.addi %body_i, %c1 : index
  cf.br ^loop(%next_i, %next_acc : index, vector<16xf32>)

^exit(%final: vector<16xf32>):
  %sink = arith.addf %final, %zero : vector<16xf32>
  return
}

// CHECK-LABEL: vc4kernel.kernel @cf_loop_vector_carry
// CHECK: cf.br
// CHECK: ^{{.*}}(%{{.*}}: i32, %{{.*}}: vector<16xf32>)
// CHECK: cf.cond_br
// CHECK: vc4kernel.fragment_alu.add
// CHECK: arith.addi
// CHECK: vc4kernel.return
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
// CHECK-NOT: scf.
// CHECK-NOT: tensor.
// CHECK-NOT: linalg.
// CHECK-NOT: gpu.
// CHECK-NOT: tt.
