// RUN: vc4-opt %s --vc4-test-print-effects -o /dev/null | FileCheck %s

// CHECK: vc4.qpu.sema: Read<Semaphore>, Write<Semaphore>

vc4.module @side_effects {
  vc4.func @scheduled_main() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    vc4.qpu.sema <release> {
      id = 2 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 0 : i32,
      waddr_mul = 0 : i32
    }
  }
}
