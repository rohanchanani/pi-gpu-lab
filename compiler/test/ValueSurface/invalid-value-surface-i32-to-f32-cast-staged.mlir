// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @i32_to_f32_cast_staged() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %v = arith.constant dense<0> : vector<16xi32>
    // expected-error @+1 {{i32 to f32 numeric cast staged by lower-half gap in Phase 14}}
    %cast = arith.sitofp %v : vector<16xi32> to vector<16xf32>
    return
  }
}
