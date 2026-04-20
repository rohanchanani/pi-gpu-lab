// RUN: vc4-opt %s --verify-diagnostics

vc4.module @bundle_requires_scheduled_form {
  vc4.func @bad_structured() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{is only legal in functions with form = scheduled}}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
    vc4.return
  }
}

vc4.module @bundle_operand_source_exclusive_error {
  vc4.func @bad_sources() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{requires exactly one of 'raddr_b' or 'small_imm'}}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  }
}

vc4.module @bundle_operand_source_double_specified_error {
  vc4.func @bad_sources() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{requires exactly one of 'raddr_b' or 'small_imm'}}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, small_imm = 2 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  }
}

vc4.module @bundle_signal_small_imm_mismatch {
  vc4.func @bad_signal() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{'small_imm' attribute requires sig = #vc4.qpu_signal<small_imm>}}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, small_imm = 2 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  }
}

vc4.module @bundle_signal_small_imm_missing_payload {
  vc4.func @bad_signal() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{sig = #vc4.qpu_signal<small_imm> requires a 'small_imm' attribute}}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  }
}

vc4.module @bundle_pack_path_error {
  vc4.func @bad_pack() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{pm = true requires 'pack' to use #vc4.mul_pack_mode}}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = true, pack = #vc4.regfile_a_pack_mode<to_16a>, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, small_imm = 2 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  }
}

vc4.module @bundle_unpack_path_error {
  vc4.func @bad_unpack() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{pm = false requires 'unpack' to use #vc4.regfile_a_unpack_mode}}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, unpack = #vc4.r4_unpack_mode<f16a>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  }
}

vc4.module @bundle_raddr_range_error {
  vc4.func @bad_raddr() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{'raddr_b' attribute must be in range [0, 63]}}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 32 : i32, raddr_b = 64 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  }
}

vc4.module @bundle_small_imm_range_error {
  vc4.func @bad_small_imm() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{'small_imm' attribute must be an encoded selector in range [0, 63]}}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, small_imm = 64 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  }
}

vc4.module @bundle_vector_rotate_requires_accumulator_mul_inputs {
  vc4.func @bad_vector_rotate() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{vector-rotate small_imm selectors 48..63 require both MUL inputs to come from accumulators r0..r3}}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<small_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, small_imm = 49 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r4>, mul_b = #vc4.qpu_mux<r1>}
  }
}

vc4.module @bundle_dedicated_signal_error {
  vc4.func @bad_sig() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{sig = #vc4.qpu_signal<load_imm> is represented by vc4.qpu.ldi}}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<load_imm>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32, op_add = #vc4.add_opcode<nop>, op_mul = #vc4.mul_opcode<nop>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  }
}

vc4.module @bundle_duplicate_accumulator_write_error {
  vc4.func @bad_write_conflict() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{active ADD and MUL pipelines must not target the same accumulator/I/O write address}}
    vc4.qpu.bundle {sig = #vc4.qpu_signal<none>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 32 : i32, waddr_mul = 32 : i32, op_add = #vc4.add_opcode<add>, op_mul = #vc4.mul_opcode<fmul>, raddr_a = 0 : i32, raddr_b = 1 : i32, add_a = #vc4.qpu_mux<a>, add_b = #vc4.qpu_mux<b>, mul_a = #vc4.qpu_mux<r0>, mul_b = #vc4.qpu_mux<r1>}
  }
}
