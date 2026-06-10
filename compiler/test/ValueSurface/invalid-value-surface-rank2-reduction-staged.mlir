// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics

builtin.module {
  func.func @rank2_reduction()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.fp_domain = "finite",
                  vc4value.reduction_policy = "finite_tree"} {
    %v = arith.constant dense<0.000000e+00> : vector<2x16xf32>
    // expected-error @+1 {{'vector.reduction' op unsupported reduction rank: 2}}
    %sum = vector.reduction <add>, %v : vector<2x16xf32> into f32
    return
  }
}
