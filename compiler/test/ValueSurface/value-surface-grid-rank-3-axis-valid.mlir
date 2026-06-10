// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @grid_rank_3_axes
  func.func @grid_rank_3_axes()
      attributes {vc4value.kernel, vc4value.grid_rank = 3 : i32} {
    // CHECK: vc4value.program_id
    %pid0 = vc4value.program_id {axis = 0 : i32} : index
    %pid1 = vc4value.program_id {axis = 1 : i32} : index
    %pid2 = vc4value.program_id {axis = 2 : i32} : index
    // CHECK: vc4value.num_programs
    %np0 = vc4value.num_programs {axis = 0 : i32} : index
    %np1 = vc4value.num_programs {axis = 1 : i32} : index
    %np2 = vc4value.num_programs {axis = 2 : i32} : index
    return
  }
}
