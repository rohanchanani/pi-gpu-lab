// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

func.func @vector_branch_condition_reject()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  // expected-note @+1 {{prior use here}}
  %mask = arith.constant dense<true> : vector<16xi1>
  // expected-error @+1 {{expects different type than prior uses: 'i1' vs 'vector<16xi1>'}}
  cf.cond_br %mask, ^then, ^else
^then:
  return
^else:
  return
}
