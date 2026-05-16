// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @kernels {
// CHECK: vc4.func @main() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<threadable>}
// CHECK: vc4.qpu.bundle
// CHECK-SAME: sig = #vc4.qpu_signal<none>
// CHECK: vc4.qpu.ldi <splat32>
// CHECK-SAME: value = 42 : i32
// CHECK: vc4.qpu.sema <release>
// CHECK-SAME: id = 1 : i32

vc4.module @kernels {
  vc4.func @main() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<threadable>} {
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 0 : i32,
      waddr_mul = 1 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 1 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
    vc4.qpu.ldi <splat32> {
      value = 42 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 2 : i32,
      waddr_mul = 3 : i32
    }
    vc4.qpu.sema <release> {
      id = 1 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 4 : i32,
      waddr_mul = 5 : i32
    }
  }
}
