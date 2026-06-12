// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_gemv_i32_policy_missing()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %a = arith.constant dense<3> : vector<16xi32>
  %x = arith.constant dense<5> : vector<16xi32>
  %prod = arith.muli %a, %x : vector<16xi32>
  %dot = vector.reduction <add>, %prod : vector<16xi32> into i32
  return
}

// CHECK: vector i32 muli requires vc4value.i32_mul_policy = "mul24_safe"
// CHECK: READY_FOR_TRITON remains NO
