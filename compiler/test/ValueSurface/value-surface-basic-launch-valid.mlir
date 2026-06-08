// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @kernel
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    // CHECK: vc4value.program_id
    %pid = vc4value.program_id {axis = 0 : i32} : index
    // CHECK: vc4value.num_programs
    %np = vc4value.num_programs {axis = 0 : i32} : index
    return
  }
}
