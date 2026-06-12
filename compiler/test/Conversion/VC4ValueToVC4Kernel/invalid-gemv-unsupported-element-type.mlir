// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_gemv_unsupported_element_type()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %a = arith.constant dense<1.000000e+00> : vector<16xf16>
  %x = arith.constant dense<2.000000e+00> : vector<16xf16>
  %prod = arith.mulf %a, %x : vector<16xf16>
  %dot = vector.reduction <add>, %prod : vector<16xf16> into f16
  return
}

// CHECK: unsupported GEMV element type
