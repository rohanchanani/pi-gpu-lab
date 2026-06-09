// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

func.func @cf_if_merge(
    %n: i32 {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : i32
  %zero = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %two = arith.constant 2.000000e+00 : f32
  %zv = vector.broadcast %zero : f32 to vector<16xf32>
  %ov = vector.broadcast %one : f32 to vector<16xf32>
  %tv = vector.broadcast %two : f32 to vector<16xf32>
  %cond = arith.cmpi sgt, %n, %c0 : i32
  // CHECK: cf.cond_br
  cf.cond_br %cond, ^then(%ov : vector<16xf32>), ^else(%tv : vector<16xf32>)

^then(%lhs: vector<16xf32>):
  %sum = arith.addf %lhs, %zv : vector<16xf32>
  cf.br ^merge(%sum : vector<16xf32>)

^else(%rhs: vector<16xf32>):
  %diff = arith.subf %rhs, %zv : vector<16xf32>
  cf.br ^merge(%diff : vector<16xf32>)

^merge(%selected: vector<16xf32>):
  // CHECK: ^{{.*}}(%{{.*}}: vector<16xf32>)
  return
}
