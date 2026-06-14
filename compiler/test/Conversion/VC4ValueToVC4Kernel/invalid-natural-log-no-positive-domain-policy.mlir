// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_natural_log_no_positive_domain_policy()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite"} {
  %x = arith.constant dense<1.000000e+00> : vector<16xf32>
  %y = math.log %x : vector<16xf32>
  return
}

// CHECK: math.log SFU mode is staged by lower-half gap or missing finite_positive domain
// CHECK: READY_FOR_TRITON remains NO
