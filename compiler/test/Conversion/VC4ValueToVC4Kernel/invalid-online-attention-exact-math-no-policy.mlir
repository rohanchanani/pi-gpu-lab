// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_online_attention_exact_math_no_policy()
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                vc4value.reduction_policy = "finite_tree",
                vc4value.max_policy = "finite"} {
  %x = arith.constant dense<0.000000e+00> : vector<16xf32>
  %e = math.exp %x : vector<16xf32>
  return
}

// CHECK: exact/default math is staged
