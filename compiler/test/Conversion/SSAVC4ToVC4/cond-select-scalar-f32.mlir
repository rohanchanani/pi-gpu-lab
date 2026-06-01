// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @cond_select_scalar_f32
// CHECK: vc4.func @kernel
// CHECK: set_flags
// CHECK: cond_add = #vc4.cond<zc>
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: ssavc4.
ssavc4.module @cond_select_scalar_f32 {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "cond_select_scalar_f32",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [
        {name = "true_value", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 0 : i32},
        {name = "false_value", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 1 : i32}
      ],
      builtins = []
    }
  } {
    %true_value = ssavc4.uniform.read 0 : f32
    %false_value = ssavc4.uniform.read 1 : f32
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %flags = ssavc4.make_flags %one, %zero {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    %selected = ssavc4.cond_select %flags, %true_value, %false_value {cond = #vc4.cond<zc>} : !ssavc4.flags, f32, f32 -> f32
    %selected_v = ssavc4.splat %selected : f32 -> vector<16xf32>
    ssavc4.thread_end
  }
}
