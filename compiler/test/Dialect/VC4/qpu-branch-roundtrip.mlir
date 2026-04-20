// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @qpu_branch {
// CHECK: vc4.func @scheduled_main() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
// CHECK: vc4.qpu.branch attributes {
// CHECK-SAME: cond = #vc4.branch_cond<any_z_clear>
// CHECK-SAME: immediate = 64 : i32
// CHECK-SAME: raddr_a = 3 : i32
// CHECK-SAME: relative = true
// CHECK-SAME: use_reg = false
// CHECK-SAME: waddr_add = 1 : i32
// CHECK-SAME: waddr_mul = 2 : i32
// CHECK: {
// CHECK: vc4.qpu.bundle
// CHECK: vc4.qpu.ldi <splat32>
// CHECK: vc4.qpu.sema <release>
// CHECK: }

vc4.module @qpu_branch {
  vc4.func @scheduled_main() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    vc4.qpu.branch attributes {
      cond = #vc4.branch_cond<any_z_clear>,
      relative = true,
      use_reg = false,
      raddr_a = 3 : i32,
      immediate = 64 : i32,
      waddr_add = 1 : i32,
      waddr_mul = 2 : i32
    } {
      vc4.qpu.bundle {
        sig = #vc4.qpu_signal<none>,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<always>,
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
        value = 7 : i32,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<never>,
        waddr_add = 2 : i32,
        waddr_mul = 3 : i32
      }
      vc4.qpu.sema <release> {
        id = 5 : i32,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<always>,
        waddr_add = 4 : i32,
        waddr_mul = 5 : i32
      }
    }
  }
}
