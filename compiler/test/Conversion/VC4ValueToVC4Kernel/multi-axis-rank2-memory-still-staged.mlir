// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @rank2_memory_still_staged(
    %out: memref<4x16xi32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32} {
  %pid1 = vc4value.program_id {axis = 1 : i32} : index
  return
}

// CHECK: expected rank-1 contiguous i32/f32 #vc4value.global memref
// CHECK: not Phase 5 lowerable
// CHECK: staged value-surface feature
// CHECK: READY_FOR_TRITON remains NO
