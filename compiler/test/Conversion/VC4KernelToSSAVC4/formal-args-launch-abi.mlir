// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @formal_args
// CHECK-SAME: args = [
// CHECK-SAME: name = "out"
// CHECK-SAME: uniform_index = 0 : i32
// CHECK-SAME: name = "n"
// CHECK-SAME: uniform_index = 1 : i32
// CHECK-SAME: name = "alpha"
// CHECK-SAME: type = "f32"
// CHECK-SAME: uniform_index = 2 : i32
// CHECK-SAME: builtins = [
// CHECK-SAME: logical_request
// CHECK-SAME: uniform_index = 3 : i32
// CHECK-SAME: total_requests
// CHECK-SAME: uniform_index = 4 : i32
// CHECK-SAME: uniform_words_per_qpu = 5 : i32
// CHECK: %[[ALPHA:.*]] = ssavc4.uniform.read 2 : f32
// CHECK: ssavc4.splat %[[ALPHA]] : f32 -> vector<16xf32>
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @formal_args(%out : i32, %n : i32, %alpha : f32) attributes {
    public_name = "formal_args",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [
      {name = "out", kind = "buffer", direction = "out", elem_type = "f32"},
      {name = "n", kind = "scalar", direction = "by_value", type = "u32"},
      {name = "alpha", kind = "scalar", direction = "by_value", type = "f32"}
    ],
    warps_per_block = 1 : i32
  } {
    %pid = vc4kernel.program_id : i32
    %alpha_v = vc4kernel.splat %alpha : f32 -> vector<16xf32>
    vc4kernel.return
  }
}
