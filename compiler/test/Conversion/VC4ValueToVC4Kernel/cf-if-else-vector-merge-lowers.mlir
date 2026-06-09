// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @cf_if_else_vector_merge(%flag: i32 {vc4value.arg_name = "flag"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : i32
  %zero = arith.constant dense<0.000000e+00> : vector<16xf32>
  %one = arith.constant dense<1.000000e+00> : vector<16xf32>
  %two = arith.constant dense<2.000000e+00> : vector<16xf32>
  %cond = arith.cmpi ne, %flag, %c0 : i32
  cf.cond_br %cond, ^then(%one : vector<16xf32>), ^else(%two : vector<16xf32>)

^then(%then_value: vector<16xf32>):
  %then_sum = arith.addf %then_value, %zero : vector<16xf32>
  cf.br ^merge(%then_sum : vector<16xf32>)

^else(%else_value: vector<16xf32>):
  %else_sum = arith.addf %else_value, %zero : vector<16xf32>
  cf.br ^merge(%else_sum : vector<16xf32>)

^merge(%selected: vector<16xf32>):
  %sink = arith.addf %selected, %zero : vector<16xf32>
  return
}

// CHECK-LABEL: vc4kernel.kernel @cf_if_else_vector_merge
// CHECK: cf.cond_br
// CHECK: ^{{.*}}(%{{.*}}: vector<16xf32>)
// CHECK: vc4kernel.fragment_alu.add
// CHECK: cf.br
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
