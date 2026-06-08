// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @launch(%out: memref<64xi32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 0 : i32} : index
  %np = vc4value.num_programs {axis = 0 : i32} : index
  return
}

// CHECK: vc4kernel.kernel @launch
// CHECK: vc4kernel.program_id
// CHECK-SAME: axis = 0
// CHECK: vc4kernel.num_programs
// CHECK-SAME: axis = 0
// CHECK: vc4kernel.return
