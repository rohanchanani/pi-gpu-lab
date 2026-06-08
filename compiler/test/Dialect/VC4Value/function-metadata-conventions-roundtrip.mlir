// RUN: vc4-opt %s | FileCheck %s

module {
  // CHECK: func.func @metadata_kernel()
  // CHECK-SAME: vc4value.grid_rank = 1 : i32
  // CHECK-SAME: vc4value.kernel
  // CHECK-SAME: vc4value.math_policy = "strict"
  // CHECK-SAME: vc4value.target_profile = "vc4"
  func.func @metadata_kernel() attributes {
    vc4value.kernel,
    vc4value.grid_rank = 1 : i32,
    vc4value.target_profile = "vc4",
    vc4value.math_policy = "strict"
  } {
    // CHECK: vc4value.program_id {axis = 0 : i32} : index
    %pid = vc4value.program_id {axis = 0 : i32} : index
    return
  }
}
