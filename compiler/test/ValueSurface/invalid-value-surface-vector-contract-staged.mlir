// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

#mat = affine_map<(m, n, k) -> (m, k)>
#vec = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>

builtin.module {
  // CHECK-LABEL: func.func @vector_contract_staged_for_phase13
  func.func @vector_contract_staged_for_phase13()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.fp_domain = "finite",
                  vc4value.reduction_policy = "finite_tree"} {
    %a = arith.constant dense<0.000000e+00> : vector<1x4xf32>
    %b = arith.constant dense<0.000000e+00> : vector<4x16xf32>
    %c = arith.constant dense<0.000000e+00> : vector<1x16xf32>
    // CHECK: vector.contract
    %contract = vector.contract {indexing_maps = [#mat, #vec, #out], iterator_types = ["parallel", "parallel", "reduction"]} %a, %b, %c : vector<1x4xf32>, vector<4x16xf32> into vector<1x16xf32>
    return
  }
}
