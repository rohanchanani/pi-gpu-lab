// RUN: vc4-opt --vc4-verify-scheduled-peripheral-accesses %s --split-input-file --verify-diagnostics

vc4.module @tmu_read_signal_and_sfu_write_same_slot {
  vc4.func @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{encodes more than one closely-coupled peripheral access in a single scheduled instruction slot (TMU read signal, SFU write)}}
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<ldtmu0>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 52 : i32,
      waddr_mul = 1 : i32,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 2 : i32,
      raddr_b = 3 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
  }
}

// -----

vc4.module @tmu_parameter_write_and_mutex_read_same_slot {
  vc4.func @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{encodes more than one closely-coupled peripheral access in a single scheduled instruction slot (TMU parameter write, mutex acquire read)}}
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 56 : i32,
      waddr_mul = 1 : i32,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 51 : i32,
      raddr_b = 4 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
  }
}

// -----

vc4.module @semaphore_and_vpm_control_access_same_slot {
  vc4.func @bad() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{encodes more than one closely-coupled peripheral access in a single scheduled instruction slot (semaphore access, VPM/VDR/VDW control register-space access)}}
    vc4.qpu.sema <release> {
      id = 3 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 49 : i32,
      waddr_mul = 1 : i32
    }
  }
}
