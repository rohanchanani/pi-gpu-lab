// RUN: vc4-opt --vc4-verify-emit-contract %s | FileCheck %s

// CHECK: vc4.module @emit_contract
// CHECK: vc4.func @kernel_entry() attributes {domain = #vc4.execution_domain<qpu>, form = #vc4.function_form<scheduled>, kernel, threading = #vc4.threading_mode<single>}
// CHECK: vc4.qpu.bundle
// CHECK: vc4.qpu.branch
// CHECK: sig = #vc4.qpu_signal<thrend>
// CHECK: vc4.qpu.ldi <splat32>
// CHECK: vc4.qpu.sema <release>
// CHECK: vc4.func @driver(%[[BASE:.*]]: i32, %[[LEN:.*]]: i32) attributes {domain = #vc4.execution_domain<host>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>}
// CHECK: %[[TOK:.*]] = "vc4.enqueue_qpu"(%[[BASE]], %[[LEN]]) <{entry = @kernel_entry}> : (i32, i32) -> !vc4.async.token
// CHECK: vc4.reserve_qpu {mask = 3 : i32}
// CHECK: vc4.async.wait %[[TOK]] : !vc4.async.token

vc4.module @emit_contract {
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
    vc4.qpu.branch attributes {
      cond = #vc4.branch_cond<always>,
      relative = true,
      use_reg = false,
      raddr_a = 0 : i32,
      immediate = 16 : i32,
      waddr_add = 0 : i32,
      waddr_mul = 0 : i32
    } {
      vc4.qpu.bundle {
        sig = #vc4.qpu_signal<none>,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<never>,
        waddr_add = 2 : i32,
        waddr_mul = 3 : i32,
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
        waddr_add = 4 : i32,
        waddr_mul = 5 : i32
      }
      vc4.qpu.sema <release> {
        id = 1 : i32,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<always>,
        waddr_add = 6 : i32,
        waddr_mul = 7 : i32
      }
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

  vc4.func @driver(%base: i32, %len: i32) attributes {domain = #vc4.execution_domain<host>, form = #vc4.function_form<structured>, threading = #vc4.threading_mode<single>} {
    %tok = "vc4.enqueue_qpu"(%base, %len) <{entry = @kernel_entry}> : (i32, i32) -> !vc4.async.token
    "vc4.reserve_qpu"() <{mask = 3 : i32}> : () -> ()
    "vc4.cf.branch"() [^bb1, ^bb2] <{cond = #vc4.branch_cond<any_z_clear>}> : () -> ()
  ^bb1:
    vc4.async.wait %tok : !vc4.async.token
    vc4.return
  ^bb2:
    vc4.return
  }
}
