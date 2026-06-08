// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @f32_cmp_without_policy(%x: f32 {vc4value.arg_name = "x"},
                                  %y: f32 {vc4value.arg_name = "y"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %xv = vector.broadcast %x : f32 to vector<16xf32>
  %yv = vector.broadcast %y : f32 to vector<16xf32>
  %cmp = arith.cmpf olt, %xv, %yv : vector<16xf32>
  return
}

// CHECK: f32 comparisons require vc4value.fp_domain = "finite"
// CHECK: not Phase 5 lowerable without explicit finite policy
// CHECK: READY_FOR_TRITON remains NO
