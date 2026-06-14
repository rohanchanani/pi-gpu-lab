// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_exact_math_no_approx_policy()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite"} {
  %x = arith.constant dense<1.000000e+00> : vector<16xf32>
  %y = math.exp %x : vector<16xf32>
  return
}

// CHECK: exact/default math requires explicit approximate-SFU policy
// CHECK: READY_FOR_TRITON remains NO
