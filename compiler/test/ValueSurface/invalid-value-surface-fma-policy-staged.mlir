// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @fma_policy_staged_for_phase13
  func.func @fma_policy_staged_for_phase13()
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                  vc4value.fp_domain = "finite",
                  vc4value.reduction_policy = "finite_tree"} {
    %a = arith.constant dense<1.000000e+00> : vector<16xf32>
    %x = arith.constant dense<2.000000e+00> : vector<16xf32>
    %acc = arith.constant dense<0.000000e+00> : vector<16xf32>
    // CHECK: math.fma
    %fused = math.fma %a, %x, %acc : vector<16xf32>
    %dot = vector.reduction <add>, %fused : vector<16xf32> into f32
    return
  }
}
