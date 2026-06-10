// RUN: vc4-opt %s --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s --check-prefix=VC4K
// RUN: vc4-opt %s --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel --convert-vc4kernel-to-ssavc4 --convert-ssavc4-to-vc4 | FileCheck %s --check-prefix=SCHED

func.func @nested_scf_if_for_while_lowers(%n: index {vc4value.arg_name = "n"},
                                          %flag: i32 {vc4value.arg_name = "flag"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zi = arith.constant 0 : i32
  %oi = arith.constant 1 : i32
  %cond = arith.cmpi ne, %flag, %zi : i32
  %outer = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zi) -> (i32) {
    %selected = scf.if %cond -> (i32) {
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

// VC4K-LABEL: vc4kernel.kernel @nested_scf_if_for_while_lowers
// VC4K: cf.br
// VC4K: ^{{.*}}(%{{.*}}: i32, %{{.*}}: i32)
// VC4K: cf.cond_br
// VC4K: arith.addi
// VC4K: vc4kernel.return
// VC4K-NOT: scf.
// VC4K-NOT: func.func
// VC4K-NOT: vector.
// VC4K-NOT: memref.
// VC4K-NOT: vc4value.
// VC4K-NOT: tt.
// VC4K-NOT: ttg.
// VC4K-NOT: gpu.

// SCHED-LABEL: vc4.func @nested_scf_if_for_while_lowers
// SCHED: vc4.qpu.branch
// SCHED: sig = #vc4.qpu_signal<thrend>
