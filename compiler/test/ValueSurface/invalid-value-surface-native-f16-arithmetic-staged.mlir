// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @native_f16_arithmetic_staged() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %a = arith.constant dense<0.000000e+00> : vector<16xf16>
    %b = arith.constant dense<1.000000e+00> : vector<16xf16>
    // expected-error @+1 {{native f16 arithmetic is not legal in the Phase 14 VC4 value surface}}
    %sum = arith.addf %a, %b : vector<16xf16>
    return
  }
}
