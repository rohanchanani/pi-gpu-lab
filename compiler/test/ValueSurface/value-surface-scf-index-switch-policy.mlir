// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

func.func @scf_index_switch_staged_reject(
    %selector: index {vc4value.arg_name = "selector"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  // expected-error @+1 {{scf operation is not in the Phase 3 VC4 value-surface control subset}}
  %result = scf.index_switch %selector -> i32
  case 0 {
    %zero = arith.constant 0 : i32
    scf.yield %zero : i32
  }
  case 1 {
    %one = arith.constant 1 : i32
    scf.yield %one : i32
  }
  default {
    %other = arith.constant 2 : i32
    scf.yield %other : i32
  }
  %sink = arith.addi %result, %result : i32
  return
}
