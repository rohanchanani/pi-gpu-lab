// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

func.func @scf_reduce_staged_reject()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  // expected-error @+1 {{scf operation is not in the Phase 3 VC4 value-surface control subset}}
  %result = scf.parallel (%i) = (%c0) to (%c1) step (%c1) init (%init) -> i32 {
    %value = arith.constant 1 : i32
    // expected-error @+1 {{scf operation is not in the Phase 3 VC4 value-surface control subset}}
      scf.reduce(%value : i32) {
    ^bb0(%lhs: i32, %rhs: i32):
      %sum = arith.addi %lhs, %rhs : i32
      // expected-error @+1 {{scf operation is not in the Phase 3 VC4 value-surface control subset}}
      scf.reduce.return %sum : i32
    }
  }
  %sink = arith.addi %result, %init : i32
  return
}
