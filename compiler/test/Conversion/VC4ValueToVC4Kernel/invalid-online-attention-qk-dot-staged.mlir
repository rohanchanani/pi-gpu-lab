// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

#vec = affine_map<(i) -> (i)>
#scalar = affine_map<(i) -> ()>

func.func @invalid_online_attention_qk_dot_staged()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %q = arith.constant dense<0.000000e+00> : vector<16xf32>
  %k = arith.constant dense<0.000000e+00> : vector<16xf32>
  %init = arith.constant 0.000000e+00 : f32
  %score = vector.contract {indexing_maps = [#vec, #vec, #scalar], iterator_types = ["reduction"]} %q, %k, %init : vector<16xf32>, vector<16xf32> into f32
  return
}

// CHECK: vector.contract is staged
