// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @native_f16_arith(%a: vector<16xf16>, %b: vector<16xf16>) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    // expected-error @+1 {{native f16 arithmetic is not legal in the Phase 3.5 VC4 value surface}}
    %sum = arith.addf %a, %b : vector<16xf16>
    return
  }
}
