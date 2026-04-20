// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @kernels {
// CHECK: vc4.func @main() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, kernel, threading = #vc4.threading_mode<threadable>}
// CHECK: %[[QPU:.*]] = vc4.builtin qpu_num : i32
// CHECK: vc4.return

vc4.module @kernels {
  vc4.func @main() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, kernel, threading = #vc4.threading_mode<threadable>} {
    %qpu = vc4.builtin qpu_num : i32
    vc4.return
  }
}
