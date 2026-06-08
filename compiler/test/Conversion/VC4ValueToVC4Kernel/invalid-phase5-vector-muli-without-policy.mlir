// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @i32_muli_without_policy()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %lhs = arith.constant dense<3> : vector<16xi32>
  %rhs = arith.constant dense<5> : vector<16xi32>
  %mul = arith.muli %lhs, %rhs : vector<16xi32>
  return
}

// CHECK: vector i32 muli requires vc4value.i32_mul_policy = "mul24_safe" in Phase 5
// CHECK: not Phase 5 lowerable
// CHECK: READY_FOR_TRITON remains NO
