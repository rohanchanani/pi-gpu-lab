// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

#mat = affine_map<(m, n, k) -> (m, k)>
#vec = affine_map<(m, n, k) -> (k, n)>
#out = affine_map<(m, n, k) -> (m, n)>

func.func @contract()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %a = arith.constant dense<1.000000e+00> : vector<1x4xf32>
  %b = arith.constant dense<2.000000e+00> : vector<4x16xf32>
  %c = arith.constant dense<0.000000e+00> : vector<1x16xf32>
  %contract = vector.contract {indexing_maps = [#mat, #vec, #out], iterator_types = ["parallel", "parallel", "reduction"]} %a, %b, %c : vector<1x4xf32>, vector<4x16xf32> into vector<1x16xf32>
  return
}

// CHECK: constant result type outside Phase 5 subset is not Phase 5 lowerable
// CHECK: staged value-surface feature
// CHECK: READY_FOR_TRITON remains NO
