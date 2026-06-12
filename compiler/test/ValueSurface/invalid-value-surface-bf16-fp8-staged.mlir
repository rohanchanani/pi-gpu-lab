// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @bf16_staged() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    // expected-error @+1 {{vector element type is not legal in the VC4 value surface}}
    %v = arith.constant dense<0.000000e+00> : vector<16xbf16>
    return
  }
}
