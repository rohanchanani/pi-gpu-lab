// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @kernels {
// CHECK: vc4.func @main() attributes {form = 0 : i32, kernel, threading = 1 : i32}
// CHECK: %[[QPU:.*]] = vc4.builtin qpu_num : i32
// CHECK: vc4.return

vc4.module @kernels {
  vc4.func @main() attributes {kernel, threading = 1 : i32, form = 0 : i32} {
    %qpu = vc4.builtin qpu_num : i32
    vc4.return
  }
}
