// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: vc4.module @kernels
// CHECK-LABEL: vc4.func @saxpy_kernel
// CHECK-SAME: kernel
// CHECK-SAME: vc4.launch_abi =
// CHECK-SAME: direction = "in"
// CHECK-SAME: elem_type = "f32"
// CHECK-SAME: kind = "buffer"
// CHECK-SAME: name = "x"
// CHECK-SAME: kind = #vc4.builtin_kind<qpu_num>
// CHECK-SAME: materialization = "uniform_suffix"
// CHECK-SAME: name = "qpu_id"
// CHECK-SAME: kind = #vc4.builtin_kind<num_qpus>
// CHECK-SAME: materialization = "uniform_suffix"
// CHECK-SAME: name = "num_qpus"
// CHECK-SAME: uniform_index = 5 : i32
// CHECK-SAME: public_name = "saxpy"
// CHECK-SAME: tail_policy = "exact_multiple"
// CHECK-SAME: uniform_words_per_qpu = 6 : i32
// CHECK-LABEL: vc4.func @uses_num_qpus
// CHECK: %{{.*}} = vc4.builtin num_qpus : i32

vc4.module @kernels {
  vc4.func @saxpy_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<structured>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "saxpy",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 6 : i32,
      args = [
        {name = "x", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 0 : i32},
        {name = "y", kind = "buffer", direction = "inout", elem_type = "f32", uniform_index = 1 : i32},
        {name = "a", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 2 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32}
      ],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 4 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 5 : i32}
      ]
    }
  } {
    vc4.return
  }

  vc4.func @uses_num_qpus() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    %0 = vc4.builtin num_qpus : i32
    vc4.return
  }
}
