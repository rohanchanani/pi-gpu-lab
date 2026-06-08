// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @non16()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c = arith.constant dense<0.000000e+00> : vector<32xf32>
  return
}

// CHECK: not Phase 5 lowerable
// CHECK: staged value-surface feature
