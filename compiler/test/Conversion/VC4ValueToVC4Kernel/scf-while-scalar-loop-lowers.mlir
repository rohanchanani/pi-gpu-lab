// RUN: vc4-opt %s --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s --check-prefix=VC4K
// RUN: vc4-opt %s --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel --convert-vc4kernel-to-ssavc4 --convert-ssavc4-to-vc4 | FileCheck %s --check-prefix=SCHED

func.func @scf_while_scalar_loop_lowers(%n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
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

// VC4K-LABEL: vc4kernel.kernel @scf_while_scalar_loop_lowers
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

// SCHED-LABEL: vc4.func @scf_while_scalar_loop_lowers
// SCHED: vc4.qpu.branch
// SCHED: sig = #vc4.qpu_signal<thrend>
