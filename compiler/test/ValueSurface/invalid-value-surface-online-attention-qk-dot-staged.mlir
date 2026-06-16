// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

#mat = affine_map<(m, n, k) -> (m, k)>
#vec = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>

builtin.module {
  // CHECK-LABEL: func.func @online_attention_qk_dot_staged
  func.func @online_attention_qk_dot_staged()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.fp_domain = "finite",
                  vc4value.reduction_policy = "finite_tree"} {
    %q = arith.constant dense<0.000000e+00> : vector<1x16xf32>
    %k = arith.constant dense<0.000000e+00> : vector<16x1xf32>
    %init = arith.constant dense<0.000000e+00> : vector<1x1xf32>
    // CHECK: vector.contract
    %score = vector.contract {indexing_maps = [#mat, #vec, #out], iterator_types = ["parallel", "parallel", "reduction"]} %q, %k, %init : vector<1x16xf32>, vector<16x1xf32> into vector<1x1xf32>
    return
  }
}
