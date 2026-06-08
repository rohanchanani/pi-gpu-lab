// RUN: vc4-opt %s | FileCheck %s

module {
  func.func @launch_ids() {
    // CHECK: vc4value.program_id {axis = 0 : i32} : index
    %pid0 = vc4value.program_id {axis = 0 : i32} : index
    // CHECK: vc4value.program_id {axis = 1 : i32} : index
    %pid1 = vc4value.program_id {axis = 1 : i32} : index
    // CHECK: vc4value.program_id {axis = 2 : i32} : index
    %pid2 = vc4value.program_id {axis = 2 : i32} : index
    // CHECK: vc4value.num_programs {axis = 0 : i32} : index
    %np0 = vc4value.num_programs {axis = 0 : i32} : index
    // CHECK: vc4value.num_programs {axis = 1 : i32} : index
    %np1 = vc4value.num_programs {axis = 1 : i32} : index
    // CHECK: vc4value.num_programs {axis = 2 : i32} : index
    %np2 = vc4value.num_programs {axis = 2 : i32} : index
    return
  }
}
