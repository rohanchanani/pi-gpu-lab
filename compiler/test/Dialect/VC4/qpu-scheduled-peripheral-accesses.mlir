// RUN: vc4-opt --vc4-verify-scheduled-peripheral-accesses %s | FileCheck %s

// CHECK: vc4.module @qpu_scheduled_peripheral_accesses
// CHECK: vc4.func @tmu_read_signal_only_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>}
// CHECK: sig = #vc4.qpu_signal<ldtmu0>
// CHECK: vc4.func @tmu_parameter_write_only_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>}
// CHECK: waddr_add = 56 : i32
// CHECK: vc4.func @semaphore_only_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>}
// CHECK: vc4.qpu.sema <release>
// CHECK: vc4.func @vpm_data_port_only_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>}
// CHECK: waddr_add = 48 : i32
// CHECK: vc4.func @mutex_read_only_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>}
// CHECK: raddr_a = 51 : i32

vc4.module @qpu_scheduled_peripheral_accesses {
  vc4.func @tmu_read_signal_only_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<ldtmu0>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 0 : i32,
      waddr_mul = 1 : i32,
      op_add = #vc4.add_opcode<nop>,
      op_mul = #vc4.mul_opcode<nop>,
      raddr_a = 2 : i32,
      raddr_b = 3 : i32,
      add_a = #vc4.qpu_mux<a>,
      add_b = #vc4.qpu_mux<b>,
      mul_a = #vc4.qpu_mux<r0>,
      mul_b = #vc4.qpu_mux<r1>
    }
  }

  vc4.func @tmu_parameter_write_only_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    vc4.qpu.ldi <splat32> {
      value = 7 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 56 : i32,
      waddr_mul = 1 : i32
    }
  }

  vc4.func @semaphore_only_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    vc4.qpu.sema <release> {
      id = 2 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 0 : i32,
      waddr_mul = 1 : i32
    }
  }

  vc4.func @vpm_data_port_only_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    vc4.qpu.sema <release> {
      id = 1 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 48 : i32,
      waddr_mul = 1 : i32
    }
  }

  vc4.func @mutex_read_only_ok() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, threading = #vc4.threading_mode<single>} {
    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<always>,
      waddr_add = 0 : i32,
      waddr_mul = 1 : i32,
      op_add = #vc4.add_opcode<nop>,
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
