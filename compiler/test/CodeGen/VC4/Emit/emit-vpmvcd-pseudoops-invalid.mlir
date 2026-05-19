// RUN: vc4-opt %s --split-input-file --verify-diagnostics

vc4.module @bad_setup_side {
  vc4.func @kernel() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    vc4.qpu.vpmvcd_setup {
      // expected-error@+2 {{expected ::mlir::vc4::VPMVCDSide to be one of: read, write}}
      // expected-error@+1 {{failed to parse VC4_VPMVCDSideAttr parameter 'value'}}
      side = #vc4.vpmvcd_side<load>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      small_imm = 0 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
  }
}

// -----

vc4.module @bad_setup_address {
  vc4.func @kernel() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{'waddr_add' must be 49 when present}}
    vc4.qpu.vpmvcd_setup {
      side = #vc4.vpmvcd_side<read>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 50 : i32,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      small_imm = 0 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
  }
}

// -----

vc4.module @bad_addr_shape {
  vc4.func @kernel() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{requires an active ADD-side write to the implied VPM/VCD/VDW control address}}
    vc4.qpu.vpmvcd_addr {
      side = #vc4.vpmvcd_side<write>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      small_imm = 0 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
  }
}

// -----

vc4.module @bad_wait_write {
  vc4.func @kernel() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{must be read-only and must not carry 'waddr_add'}}
    vc4.qpu.vpmvcd_wait {
      side = #vc4.vpmvcd_side<read>,
      waddr_add = 32 : i32
    }
  }
}

// -----

vc4.module @bad_setup_ldi_address {
  vc4.func @kernel() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{'waddr_add' must be 49 when present}}
    vc4.qpu.vpmvcd_setup_ldi <splat32> {
      side = #vc4.vpmvcd_side<write>,
      value = -2138030080 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 50 : i32
    }
  }
}

// -----

vc4.module @bad_raw_ldi_setup {
  vc4.func @kernel() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{raw vc4.qpu.ldi writes to VPM/VCD/VDW setup address 49 require vc4.qpu.vpmvcd_setup or vc4.qpu.vpmvcd_setup_ldi}}
    vc4.qpu.ldi <splat32> {value = -2138030080 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 49 : i32, waddr_mul = 32 : i32}
  }
}

// -----

vc4.module @bad_raw_bundle_setup {
  vc4.func @kernel() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{raw vc4.qpu.bundle writes to VPM/VCD/VDW setup address 49 require vc4.qpu.vpmvcd_setup or vc4.qpu.vpmvcd_setup_ldi}}
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 49 : i32,
      waddr_mul = 32 : i32,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      small_imm = 0 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
  }
}

// -----

vc4.module @bad_raw_bundle_addr {
  vc4.func @kernel() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{raw vc4.qpu.bundle writes to VPM/VCD/VDW address register 50 require vc4.qpu.vpmvcd_addr}}
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 50 : i32,
      waddr_mul = 32 : i32,
      op_add = #vc4.add_opcode<add>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      small_imm = 0 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
  }
}

// -----

vc4.module @bad_raw_bundle_wait {
  vc4.func @kernel() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{raw vc4.qpu.bundle reads from VPM/VCD/VDW wait address 50 require vc4.qpu.vpmvcd_wait}}
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<never>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 50 : i32,
      raddr_b = 1 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
  }
}

// -----

vc4.module @raw_vpm_data_port_is_allowed {
  vc4.func @kernel() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 48 : i32,
      waddr_mul = 32 : i32,
      op_add = #vc4.add_opcode<or>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 0 : i32,
      raddr_b = 1 : i32,
      add_a = #vc4.qpu_mux<r0>,
      add_b = #vc4.qpu_mux<r0>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
  }
}
