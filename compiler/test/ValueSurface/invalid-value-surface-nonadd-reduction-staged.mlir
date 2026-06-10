// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @nonadd_reduction()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %v = arith.constant dense<1> : vector<16xi32>
    // expected-error @+1 {{Phase 12 value reductions only accept vector.reduction <add>; non-add reductions are staged}}
    %sum = vector.reduction <mul>, %v : vector<16xi32> into i32
    return
  }
}
