// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

func.func @scf_parallel_staged_reject()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @+1 {{scf operation is not in the Phase 3 VC4 value-surface control subset}}
  scf.parallel (%i) = (%c0) to (%c1) step (%c1) {
  }
  return
}
