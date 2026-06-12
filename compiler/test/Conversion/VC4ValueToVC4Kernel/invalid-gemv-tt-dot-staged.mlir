// RUN: not vc4-opt --allow-unregistered-dialect %s --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_gemv_tt_dot_staged()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %a = arith.constant dense<1.000000e+00> : vector<16xf32>
  %b = arith.constant dense<2.000000e+00> : vector<16xf32>
  %acc = arith.constant dense<0.000000e+00> : vector<16xf32>
  %dot = "tt.dot"(%a, %b, %acc) : (vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>
  return
}

// CHECK: tt.dot is staged
// CHECK: READY_FOR_TRITON remains NO
