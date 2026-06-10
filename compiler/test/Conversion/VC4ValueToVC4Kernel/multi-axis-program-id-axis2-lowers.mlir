// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @pid_axis2()
    attributes {vc4value.kernel, vc4value.grid_rank = 3 : i32} {
  %pid2 = vc4value.program_id {axis = 2 : i32} : index
  return
}

// CHECK-LABEL: vc4kernel.kernel @pid_axis2
// CHECK: vc4kernel.program_id
// CHECK-SAME: axis = 2
// CHECK: vc4kernel.return
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
