// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

func.func @scf_while_scalar_surface(
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  // CHECK: scf.while
  // CHECK: scf.condition
  %result:2 = scf.while (%i = %c0, %acc = %zero) : (index, i32) -> (index, i32) {
    %keep_going = arith.cmpi ult, %i, %n : index
    scf.condition(%keep_going) %i, %acc : index, i32
  } do {
  ^bb0(%body_i: index, %body_acc: i32):
    %next_i = arith.addi %body_i, %c1 : index
    %next_acc = arith.addi %body_acc, %one : i32
    scf.yield %next_i, %next_acc : index, i32
  }
  %sink = arith.addi %result#1, %zero : i32
  return
}

// -----

func.func @scf_while_vector_surface(
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant dense<0> : vector<16xi32>
  %one = arith.constant dense<1> : vector<16xi32>
  // CHECK: scf.while
  %result:2 = scf.while (%i = %c0, %acc = %zero) : (index, vector<16xi32>) -> (index, vector<16xi32>) {
    %keep_going = arith.cmpi ult, %i, %n : index
    scf.condition(%keep_going) %i, %acc : index, vector<16xi32>
  } do {
  ^bb0(%body_i: index, %body_acc: vector<16xi32>):
    %next_i = arith.addi %body_i, %c1 : index
    %next_acc = arith.addi %body_acc, %one : vector<16xi32>
    scf.yield %next_i, %next_acc : index, vector<16xi32>
  }
  %sink = arith.addi %result#1, %zero : vector<16xi32>
  return
}
