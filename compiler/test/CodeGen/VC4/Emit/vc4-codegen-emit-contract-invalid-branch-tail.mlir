// RUN: rm -rf %t.bundle
// RUN: not vc4-codegen %s --emit-bundle %t.bundle 2>&1 | FileCheck %s

// CHECK: qasm input requires an explicit thrend plus two delay-slot instructions
// CHECK: slot N-3 must be a vc4.qpu.bundle with sig = #vc4.qpu_signal<thrend>

// The branch itself is verifier-clean and has exactly three delay-slot ops; it
// is still not a valid qasm epilogue tail after the explicit thread-end slot.

vc4.module @vc4_codegen_emit_contract_branch_tail {
  vc4.func @branch_tail_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "branch_tail_launch",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    }
  } {
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
  }
}
