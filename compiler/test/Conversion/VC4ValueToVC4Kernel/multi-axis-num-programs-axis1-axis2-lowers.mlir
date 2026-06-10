// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @num_programs_axes()
    attributes {vc4value.kernel, vc4value.grid_rank = 3 : i32} {
  %n1 = vc4value.num_programs {axis = 1 : i32} : index
  %n2 = vc4value.num_programs {axis = 2 : i32} : index
  return
}

// CHECK-LABEL: vc4kernel.kernel @num_programs_axes
// CHECK: vc4kernel.num_programs
// CHECK-SAME: axis = 1
// CHECK: vc4kernel.num_programs
// CHECK-SAME: axis = 2
// CHECK: vc4kernel.return
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
