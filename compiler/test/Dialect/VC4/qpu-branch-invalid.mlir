// RUN: vc4-opt %s --verify-diagnostics

vc4.module @branch_requires_scheduled_form {
  // expected-error@+1 {{structured vc4 form has been removed; use ssavc4 for pre-scheduled SSA IR}}
  vc4.func @bad_structured() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    %0 = "builtin.unrealized_conversion_cast"() : () -> i32
  }
}

vc4.module @branch_zero_delay_slots {
  vc4.func @bad_zero_slots() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{delay-slot region must contain exactly 3 scheduled QPU ops}}
    vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_z_set>, relative = false, use_reg = true, raddr_a = 1 : i32, immediate = 8 : i32, waddr_add = 0 : i32, waddr_mul = 0 : i32} {
    }
  }
}

vc4.module @branch_two_delay_slots {
  vc4.func @bad_two_slots() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{delay-slot region must contain exactly 3 scheduled QPU ops}}
    vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_z_set>, relative = false, use_reg = true, raddr_a = 1 : i32, immediate = 8 : i32, waddr_add = 0 : i32, waddr_mul = 0 : i32} {
      vc4.qpu.ldi <splat32> {value = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
      vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
    }
  }
}

vc4.module @branch_four_delay_slots {
  vc4.func @bad_four_slots() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{delay-slot region must contain exactly 3 scheduled QPU ops}}
    vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_z_set>, relative = false, use_reg = true, raddr_a = 1 : i32, immediate = 8 : i32, waddr_add = 0 : i32, waddr_mul = 0 : i32} {
      vc4.qpu.ldi <splat32> {value = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
      vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
      vc4.qpu.ldi <splat32> {value = 2 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 1 : i32, waddr_mul = 1 : i32}
    }
  }
}

vc4.module @branch_delay_slot_mixing_error {
  vc4.func @bad_delay_slot_mixing() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_z_clear>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = 4 : i32, waddr_add = 0 : i32, waddr_mul = 0 : i32} {
      // expected-error@+1 {{is not legal in scheduled VC4 functions; expected a vc4.qpu.* op}}
      %0 = "builtin.unrealized_conversion_cast"() : () -> i32
      vc4.qpu.ldi <splat32> {value = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
      vc4.qpu.sema <acquire> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
    }
  }
}

vc4.module @branch_multiple_blocks_error {
  vc4.func @bad_multiple_blocks() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{expects region #0 to have 0 or 1 blocks}}
    vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_z_set>, relative = false, use_reg = false, raddr_a = 0 : i32, immediate = 12 : i32, waddr_add = 0 : i32, waddr_mul = 0 : i32} {
    ^bb0:
      vc4.qpu.ldi <splat32> {value = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
    ^bb1:
      vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    }
  }
}

vc4.module @branch_block_arguments_error {
  vc4.func @bad_block_arguments() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{delay-slot region block must not take arguments}}
    vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_z_set>, relative = true, use_reg = false, raddr_a = 0 : i32, immediate = 16 : i32, waddr_add = 0 : i32, waddr_mul = 0 : i32} {
    ^bb0(%slot: i32):
      vc4.qpu.ldi <splat32> {value = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
      vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    }
  }
}

vc4.module @branch_raddr_range_error {
  vc4.func @bad_raddr() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{'raddr_a' attribute must be in range [0, 31]}}
    vc4.qpu.branch attributes {cond = #vc4.branch_cond<always>, relative = false, use_reg = true, raddr_a = 32 : i32, immediate = 0 : i32, waddr_add = 0 : i32, waddr_mul = 0 : i32} {
      vc4.qpu.ldi <splat32> {value = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
      vc4.qpu.sema <release> {id = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
      vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    }
  }
}
