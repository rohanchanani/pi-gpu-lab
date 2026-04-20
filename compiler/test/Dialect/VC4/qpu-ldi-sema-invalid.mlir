// RUN: vc4-opt %s --verify-diagnostics

vc4.module @ldi_requires_scheduled_form {
  vc4.func @bad_structured() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{is only legal in functions with form = scheduled}}
    vc4.qpu.ldi <splat32> {value = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
    vc4.return
  }
}

vc4.module @ldi_splat_payload_error {
  vc4.func @bad_payload() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{splat32 mode requires a signless i32 'value' attribute}}
    vc4.qpu.ldi <splat32> {value = array<i32: 0, 1, 2, 3>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
  }
}

vc4.module @ldi_per_elem_range_error {
  vc4.func @bad_lanes() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{lane values for mode per_elem_i2 must be in range [-2, 1]}}
    vc4.qpu.ldi <per_elem_i2> {value = array<i32: -2, -1, 0, 1, 2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1>, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
  }
}

vc4.module @ldi_pack_path_error {
  vc4.func @bad_pack() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{pm = false requires 'pack' to use #vc4.regfile_a_pack_mode}}
    vc4.qpu.ldi <splat32> {value = 1 : i32, pm = false, pack = #vc4.mul_pack_mode<to_8a>, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
  }
}

vc4.module @ldi_waddr_error {
  vc4.func @bad_waddr() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{'waddr_add' attribute must be in range [0, 63]}}
    vc4.qpu.ldi <splat32> {value = 1 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 64 : i32, waddr_mul = 0 : i32}
  }
}

vc4.module @sema_id_error {
  vc4.func @bad_id() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{'id' attribute must be in range [0, 15]}}
    vc4.qpu.sema <release> {id = 16 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
  }
}

vc4.module @sema_pack_path_error {
  vc4.func @bad_pack() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{pm = true requires 'pack' to use #vc4.mul_pack_mode}}
    vc4.qpu.sema <acquire> {id = 2 : i32, pm = true, pack = #vc4.regfile_a_pack_mode<to_16a>, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 0 : i32}
  }
}

vc4.module @sema_stall_capable_write_address_error {
  vc4.func @bad_waddr() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    // expected-error@+1 {{'waddr_mul' must not target stall-capable peripheral write addresses}}
    vc4.qpu.sema <release> {id = 2 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<always>, waddr_add = 0 : i32, waddr_mul = 56 : i32}
  }
}
