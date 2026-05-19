// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernels/vpmvcd_setup_ldi_pseudoops.qasm
// RUN: FileCheck %s --check-prefix=QASM --input-file=%t.bundle/kernels/vpmvcd_setup_ldi_pseudoops.qasm

// QASM: ldi vr_setup, 0x00101a00
// QASM-NEXT: ldi vw_setup, 0x80904000
// QASM-NEXT: thrend
// QASM-NEXT: nop
// QASM-NEXT: nop

vc4.module @vpmvcd_setup_ldi_pseudoops {
  vc4.func @kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vpmvcd_setup_ldi_pseudoops",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    }
  } {
    vc4.qpu.vpmvcd_setup_ldi <splat32> {
      side = #vc4.vpmvcd_side<read>,
      value = 1055232 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 49 : i32
    }

    vc4.qpu.vpmvcd_setup_ldi <splat32> {
      side = #vc4.vpmvcd_side<write>,
      value = -2138030080 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 49 : i32
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

    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
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
  }
}
