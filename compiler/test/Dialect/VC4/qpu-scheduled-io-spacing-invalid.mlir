// RUN: vc4-opt --vc4-verify-scheduled-io-spacing %s --split-input-file --verify-diagnostics

vc4.module @uniform_read_too_soon_after_uniforms_address {
  vc4.func @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    vc4.qpu.ldi <splat32> {
      value = 1 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 40 : i32,
      waddr_mul = 33 : i32
    }

    vc4.qpu.sema <release> {
      id = 1 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 34 : i32,
      waddr_mul = 35 : i32
    }

    // expected-error@+1 {{reads uniform register-space address 32 within two instruction slots after a write to UNIFORMS_ADDRESS 40}}
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 36 : i32,
      waddr_mul = 37 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 32 : i32,
      raddr_b = 2 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
  }
}

// -----

vc4.module @tmu_parameter_write_too_soon_after_noswap {
  vc4.func @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    vc4.qpu.ldi <splat32> {
      value = 2 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 36 : i32,
      waddr_mul = 33 : i32
    }

    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 34 : i32,
      waddr_mul = 35 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 3 : i32,
      raddr_b = 4 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }

    // expected-error@+1 {{writes TMU parameter register-space addresses 56..63 only 2 instruction slot(s) after a write to TMU_NOSWAP 36; the first later TMU parameter write must be at least three instruction slots later}}
    vc4.qpu.ldi <splat32> {
      value = 3 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 56 : i32,
      waddr_mul = 39 : i32
    }
  }
}
