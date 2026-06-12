// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @f16_reduction_staged() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %v = arith.constant dense<0.000000e+00> : vector<16xf16>
    // expected-error @+1 {{Phase 12 value reductions only accept i32 and f32 element types}}
    %sum = vector.reduction <add>, %v : vector<16xf16> into f16
    return
  }
}
