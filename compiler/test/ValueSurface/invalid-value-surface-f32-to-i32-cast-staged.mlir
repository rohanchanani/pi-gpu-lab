// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @f32_to_i32_cast_staged() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %v = arith.constant dense<0.000000e+00> : vector<16xf32>
    // expected-error @+1 {{fp-to-int numeric casts are staged in the Phase 14 VC4 value surface}}
    %cast = arith.fptosi %v : vector<16xf32> to vector<16xi32>
    return
  }
}
