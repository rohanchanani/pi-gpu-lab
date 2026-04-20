// RUN: vc4-opt --vc4-verify-scheduled-hardware-rules %s --split-input-file --verify-diagnostics

vc4.module @thread_end_must_not_write_physical_regfile {
  vc4.func @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<threadable>} {
    // expected-error@+1 {{sig = #vc4.qpu_signal<thrend> must not write physical regfile A/B addresses 0..31}}
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<thrend>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 1 : i32,
      waddr_mul = 32 : i32,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 1 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
  }
}

// -----

vc4.module @thread_end_window_rejects_register_14 {
  vc4.func @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<threadable>} {
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

    // expected-error@+1 {{is in the thread-end hazard window and must not read or write regfile address 14}}
    vc4.qpu.ldi <splat32> {
      value = 1 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 14 : i32,
      waddr_mul = 35 : i32
    }

    vc4.qpu.sema <release> {
      id = 1 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 36 : i32,
      waddr_mul = 37 : i32
    }
  }
}

// -----

vc4.module @thread_end_window_rejects_vpm_vdr_vdw_access {
  vc4.func @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<threadable>} {
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

    vc4.qpu.branch attributes {
      cond = #vc4.branch_cond<always>,
      relative = false,
      use_reg = true,
      raddr_a = 3 : i32,
      immediate = 0 : i32,
      waddr_add = 34 : i32,
      waddr_mul = 35 : i32
    } {
      // expected-error@+1 {{is in the thread-end hazard window and must not access uniforms or VPM/VDR/VDW register-space addresses}}
      vc4.qpu.ldi <splat32> {
        value = 7 : i32,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<never>,
        waddr_add = 48 : i32,
        waddr_mul = 32 : i32
      }
      vc4.qpu.sema <release> {
        id = 1 : i32,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<always>,
        waddr_add = 36 : i32,
        waddr_mul = 37 : i32
      }
      vc4.qpu.bundle {
        sig = #vc4.qpu_signal<none>,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<always>,
        waddr_add = 38 : i32,
        waddr_mul = 39 : i32,
        op_add = #vc4.add_opcode<nop>,
        op_mul = #vc4.mul_opcode<nop>,
        raddr_a = 4 : i32,
        raddr_b = 5 : i32,
        add_a = #vc4.qpu_mux<a>,
        add_b = #vc4.qpu_mux<b>,
        mul_a = #vc4.qpu_mux<r0>,
        mul_b = #vc4.qpu_mux<r1>
      }
    }
  }
}

// -----

vc4.module @last_thread_switch_requires_threadable {
  vc4.func @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{sig = #vc4.qpu_signal<last_thread_switch> is only legal in functions with threading = #vc4.threading_mode<threadable>}}
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
      value = 1 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 34 : i32,
      waddr_mul = 35 : i32
    }

    vc4.qpu.sema <release> {
      id = 1 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 36 : i32,
      waddr_mul = 37 : i32
    }
  }
}
