// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @rank2(%x: memref<4x16xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  return
}

// CHECK: expected rank-1 contiguous i32/f32/f16 #vc4value.global memref
// CHECK: not Phase 5 lowerable
// CHECK: staged value-surface feature
