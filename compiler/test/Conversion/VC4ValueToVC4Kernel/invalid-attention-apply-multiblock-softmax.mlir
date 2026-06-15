// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_attention_apply_multiblock_softmax()
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                vc4value.attention_apply_v0 = "multiblock",
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite"} {
  return
}

// CHECK: online/multiblock attention-apply softmax is staged in Phase 16
