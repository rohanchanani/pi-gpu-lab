// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @axis1()
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %pid = vc4value.program_id {axis = 1 : i32} : index
  return
}

// CHECK: program_id axis 1 is outside vc4value.grid_rank 1
// CHECK: logical grid rank
// CHECK: READY_FOR_TRITON remains NO
