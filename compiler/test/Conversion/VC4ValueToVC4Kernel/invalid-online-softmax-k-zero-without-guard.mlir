// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_online_softmax_k_zero_without_guard()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.softmax_v0 = "zero_active",
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite"} {
  return
}

// CHECK: active-count-zero softmax is staged without an explicit finite no-op guard in Phase 15
