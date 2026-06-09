// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

func.func @cf_loop(
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %acc0 = vector.broadcast %zero : f32 to vector<16xf32>
  %step = vector.broadcast %one : f32 to vector<16xf32>
  // CHECK: cf.br
  cf.br ^loop(%c0, %acc0 : index, vector<16xf32>)

^loop(%i: index, %acc: vector<16xf32>):
  %done = arith.cmpi uge, %i, %n : index
  // CHECK: cf.cond_br
  cf.cond_br %done, ^exit(%acc : vector<16xf32>), ^body(%i, %acc : index, vector<16xf32>)

^body(%i_body: index, %acc_body: vector<16xf32>):
  %next_i = arith.addi %i_body, %c1 : index
  %next_acc = arith.addf %acc_body, %step : vector<16xf32>
  cf.br ^loop(%next_i, %next_acc : index, vector<16xf32>)

^exit(%final: vector<16xf32>):
  return
}
