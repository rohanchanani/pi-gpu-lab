// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_nan_inf_exact_semantics()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "nan_inf"} {
  %x = arith.constant dense<1.000000e+00> : vector<16xf32>
  %y = math.exp %x : vector<16xf32>
  return
}

// CHECK: NaN/Inf exact math semantics are staged
// CHECK: READY_FOR_TRITON remains NO
