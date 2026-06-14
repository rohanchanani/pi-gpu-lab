// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_generic_divf_no_approx_policy()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite_nonzero"} {
  %num = arith.constant dense<1.000000e+00> : vector<16xf32>
  %den = arith.constant dense<2.000000e+00> : vector<16xf32>
  %y = arith.divf %num, %den : vector<16xf32>
  return
}

// CHECK: generic division without approximate reciprocal policy is staged
// CHECK: READY_FOR_TRITON remains NO
