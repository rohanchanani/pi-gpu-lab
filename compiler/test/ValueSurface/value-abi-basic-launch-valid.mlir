// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @basic_launch
  func.func @basic_launch(
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %alpha: f32 {vc4value.arg_name = "alpha"},
      %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x",
                                           vc4value.direction = "in"},
      %y: memref<?xf32, #vc4value.global> {vc4value.arg_name = "y",
                                           vc4value.direction = "inout"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    // CHECK: vc4value.program_id
    %pid = vc4value.program_id {axis = 0 : i32} : index
    // CHECK: vc4value.num_programs
    %np = vc4value.num_programs {axis = 0 : i32} : index
    func.return
  }
}
