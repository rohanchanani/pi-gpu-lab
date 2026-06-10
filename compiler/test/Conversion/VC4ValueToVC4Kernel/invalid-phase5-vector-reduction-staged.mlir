// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @reduction()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree"} {
  %v = arith.constant dense<1.000000e+00> : vector<16xf32>
  %sum = vector.reduction <add>, %v : vector<16xf32> into f32
  return
}

// CHECK: vector operation 'vector.reduction' is not Phase 5 lowerable
// CHECK: staged value-surface feature
// CHECK: READY_FOR_TRITON remains NO
