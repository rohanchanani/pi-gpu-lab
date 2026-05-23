// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.func @formal_args_lowering
// CHECK-SAME: name = "out"
// CHECK-SAME: name = "n"
// CHECK-SAME: name = "alpha"
// CHECK-SAME: uniform_words_per_qpu = 3 : i32
// CHECK: ssavc4.uniform.read 0 : i32
// CHECK: ssavc4.uniform.read 1 : i32
// CHECK: ssavc4.uniform.read 2 : f32
vc4tile.kernel @formal_args_lowering attributes {
  public_name = "formal_args_lowering",
  arg_attrs = [
    {abi_name = "out", direction = "out", elem_type = "f32", kind = "buffer", type = "u32"},
    {abi_name = "n", direction = "by_value", kind = "scalar", type = "u32"},
    {abi_name = "alpha", direction = "by_value", kind = "scalar", type = "f32"}
  ]
} {
^entry(%out: i32, %n: i32, %alpha: f32):
  vc4tile.return
}

// CHECK-LABEL: ssavc4.func @formal_args_with_program_id
// CHECK-SAME: name = "out"
// CHECK-SAME: #vc4.builtin_kind<logical_request>
// CHECK-SAME: #vc4.builtin_kind<total_requests>
// CHECK-SAME: uniform_words_per_qpu = 3 : i32
// CHECK: ssavc4.uniform.read 0 : i32
// CHECK: ssavc4.uniform.read 1 : i32
vc4tile.kernel @formal_args_with_program_id attributes {
  public_name = "formal_args_with_program_id",
  arg_attrs = [{abi_name = "out", direction = "out", elem_type = "u32", kind = "buffer", type = "u32"}]
} {
^entry(%out: i32):
  %pid = vc4tile.program_id : i32
  vc4tile.return
}

// CHECK-LABEL: ssavc4.func @zero_formal_still_lowers
// CHECK-SAME: args = []
// CHECK-SAME: builtins = []
// CHECK-SAME: uniform_words_per_qpu = 0 : i32
// CHECK-NOT: ssavc4.uniform.read
// CHECK: ssavc4.thread_end
vc4tile.kernel @zero_formal_still_lowers attributes {
  public_name = "zero_formal_still_lowers"
} {
  vc4tile.return
}
