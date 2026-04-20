// RUN: vc4-opt --vc4-verify-emit-contract %s --split-input-file --verify-diagnostics

vc4.module @qpu_structured_not_directly_emittable {
  // expected-error@+1 {{is not directly emittable: qasm emission later consumes only domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled> functions}}
  vc4.func @kernel_entry() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, kernel, threading = #vc4.threading_mode<threadable>} {
    %0 = vc4.uniform.read : i32
    "vc4.semaphore"() <{id = 2 : i32, mode = #vc4.semaphore_mode<release>}> : () -> ()
    vc4.program_end
    vc4.return
  }
}

// -----

vc4.module @host_scheduled_not_directly_emittable {
  // Generic syntax keeps the function non-external without introducing
  // scheduled-body ops that would trigger unrelated dialect verifiers first.
  // expected-error@+1 {{'vc4.func' op is not directly emittable: launcher generation later consumes only domain = #vc4.execution_domain<host>, form = #vc4.function_form<structured> functions}}
  "vc4.func"() <{domain = #vc4.execution_domain<host>, form = #vc4.function_form<scheduled>, function_type = () -> (), sym_name = "driver_stub", sym_visibility = "private", threading = #vc4.threading_mode<single>}> ({
  ^bb0:
  }) : () -> ()
}

// -----

vc4.module @qpu_scheduled_missing_thrend_epilogue {
  // expected-error@+1 {{'vc4.func' op is not directly emittable: qasm input requires an explicit thrend plus two delay-slot instructions at the end of the flattened scheduled instruction stream; slot N-3 must be a vc4.qpu.bundle with sig = #vc4.qpu_signal<thrend>}}
  vc4.func @kernel_entry() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>} {
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
      value = 1 : i32,
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

// -----

vc4.module @qpu_scheduled_duplicate_thrend_epilogue {
  // expected-error@+1 {{'vc4.func' op is not directly emittable: qasm input requires an explicit thrend plus two delay-slot instructions at the end of the flattened scheduled instruction stream; found an earlier vc4.qpu.bundle with sig = #vc4.qpu_signal<thrend> before slot N-3}}
  vc4.func @kernel_entry() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>} {
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
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<thrend>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 2 : i32,
      raddr_b = 3 : i32,
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
    vc4.qpu.sema <release> {
      id = 2 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 36 : i32,
      waddr_mul = 37 : i32
    }
  }
}

// -----

vc4.module @qpu_scheduled_thrend_not_in_slot_n_minus_3 {
  // expected-error@+1 {{'vc4.func' op is not directly emittable: qasm input requires an explicit thrend plus two delay-slot instructions at the end of the flattened scheduled instruction stream; slot N-3 must be a vc4.qpu.bundle with sig = #vc4.qpu_signal<thrend>}}
  vc4.func @kernel_entry() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>} {
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
      value = 3 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 2 : i32,
      waddr_mul = 3 : i32
    }
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<thrend>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 2 : i32,
      raddr_b = 3 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
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
}

// -----

vc4.module @qpu_scheduled_epilogue_tail_must_not_branch {
  // expected-error@+1 {{'vc4.func' op is not directly emittable: qasm input requires an explicit thrend plus two delay-slot instructions at the end of the flattened scheduled instruction stream; slot N-3 must be a vc4.qpu.bundle with sig = #vc4.qpu_signal<thrend>}}
  vc4.func @kernel_entry() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>} {
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
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<thrend>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 2 : i32,
      raddr_b = 3 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
    vc4.qpu.branch attributes {
      cond = #vc4.branch_cond<always>,
      relative = true,
      use_reg = false,
      raddr_a = 0 : i32,
      immediate = 16 : i32,
      waddr_add = 4 : i32,
      waddr_mul = 5 : i32
    } {
      vc4.qpu.bundle {
        sig = #vc4.qpu_signal<none>,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<never>,
        waddr_add = 6 : i32,
        waddr_mul = 7 : i32,
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
        waddr_add = 8 : i32,
        waddr_mul = 9 : i32
      }
      vc4.qpu.sema <release> {
        id = 3 : i32,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<always>,
        waddr_add = 10 : i32,
        waddr_mul = 11 : i32
      }
    }
  }
}

// -----

vc4.module @qpu_scheduled_stream_too_short_for_epilogue {
  // expected-error@+1 {{'vc4.func' op is not directly emittable: qasm input requires an explicit thrend plus two delay-slot instructions at the end of the flattened scheduled instruction stream; found only 2 scheduled instruction slot(s)}}
  vc4.func @kernel_entry() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>} {
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

vc4.module @qpu_scheduled_duplicate_thrend_in_tail_delay_slots {
  // expected-error@+1 {{'vc4.func' op is not directly emittable: qasm input requires an explicit thrend plus two delay-slot instructions at the end of the flattened scheduled instruction stream; only slot N-3 may carry sig = #vc4.qpu_signal<thrend>; slots N-2 and N-1 must be non-branch scheduled ops without another thread-end signal}}
  vc4.func @kernel_entry() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>} {
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
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<thrend>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 2 : i32,
      raddr_b = 3 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<thrend>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 34 : i32,
      waddr_mul = 35 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 4 : i32,
      raddr_b = 5 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
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
}
