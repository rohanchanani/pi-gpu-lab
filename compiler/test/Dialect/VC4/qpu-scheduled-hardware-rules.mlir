// RUN: vc4-opt --vc4-verify-scheduled-hardware-rules %s | FileCheck %s

// CHECK: vc4.module @qpu_scheduled_hardware_rules
// CHECK: vc4.func @scheduled_thread_end_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<threadable>}
// CHECK: sig = #vc4.qpu_signal<thrend>
// CHECK: vc4.qpu.ldi <splat32>
// CHECK: vc4.qpu.sema <release>
// CHECK: vc4.func @scheduled_last_thread_switch_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<threadable>}
// CHECK: sig = #vc4.qpu_signal<last_thread_switch>
// CHECK: vc4.qpu.branch
// CHECK: vc4.func @scheduled_thread_end_mutex_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<threadable>}

vc4.module @qpu_scheduled_hardware_rules {
  vc4.func @scheduled_thread_end_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<threadable>} {
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<thrend>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32,
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
      value = 11 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 34 : i32,
      waddr_mul = 35 : i32
    }

    vc4.qpu.sema <release> {
      id = 2 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 36 : i32,
      waddr_mul = 37 : i32
    }
  }

  vc4.func @scheduled_last_thread_switch_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<threadable>} {
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<last_thread_switch>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32,
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
      waddr_add = 34 : i32,
      waddr_mul = 35 : i32
    }

    vc4.qpu.sema <release> {
      id = 3 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 36 : i32,
      waddr_mul = 37 : i32
    }

    vc4.qpu.branch attributes {
      cond = #vc4.branch_cond<always>,
      relative = false,
      use_reg = true,
      raddr_a = 3 : i32,
      immediate = 0 : i32,
      waddr_add = 38 : i32,
      waddr_mul = 39 : i32
    } {
      vc4.qpu.bundle {
        sig = #vc4.qpu_signal<none>,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<always>,
        waddr_add = 32 : i32,
        waddr_mul = 33 : i32,
        op_add = #vc4.add_opcode<nop>,
        op_mul = #vc4.mul_opcode<nop>,
        raddr_a = 4 : i32,
        raddr_b = 5 : i32,
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
        waddr_add = 34 : i32,
        waddr_mul = 35 : i32
      }
      vc4.qpu.sema <acquire> {
        id = 1 : i32,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<always>,
        waddr_add = 36 : i32,
        waddr_mul = 37 : i32
      }
    }
  }

  vc4.func @scheduled_thread_end_mutex_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<threadable>} {
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<thrend>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32,
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
      value = 9 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 51 : i32,
      waddr_mul = 34 : i32
    }

    vc4.qpu.sema <release> {
      id = 0 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 36 : i32,
      waddr_mul = 37 : i32
    }
  }
}
