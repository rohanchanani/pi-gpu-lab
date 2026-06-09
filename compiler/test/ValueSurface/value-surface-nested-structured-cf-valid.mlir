// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

func.func @nested_structured_cf_surface(
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
    %flag: i32 {vc4value.arg_name = "flag"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zi = arith.constant 0 : i32
  %oi = arith.constant 1 : i32
  %cond = arith.cmpi ne, %flag, %zi : i32
  // CHECK: scf.for
  %outer = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zi) -> (i32) {
    // CHECK: scf.if
    %selected = scf.if %cond -> (i32) {
      // CHECK: scf.while
      %inner:2 = scf.while (%j = %c0, %wacc = %acc) : (index, i32) -> (index, i32) {
        %keep_going = arith.cmpi ult, %j, %n : index
        scf.condition(%keep_going) %j, %wacc : index, i32
      } do {
      ^bb0(%body_j: index, %body_acc: i32):
        %next_j = arith.addi %body_j, %c1 : index
        %next_acc = arith.addi %body_acc, %oi : i32
        scf.yield %next_j, %next_acc : index, i32
      }
      scf.yield %inner#1 : i32
    } else {
      %next = arith.addi %acc, %oi : i32
      scf.yield %next : i32
    }
    scf.yield %selected : i32
  }
  %sink = arith.addi %outer, %zi : i32
  return
}
