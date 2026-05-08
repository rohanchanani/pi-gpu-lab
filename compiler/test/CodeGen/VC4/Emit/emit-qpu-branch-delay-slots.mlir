// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernel.qasm
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=QASM --input-file=%t.bundle/kernel.qasm
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json

// QASM: :vc4_qpu_slot_0
// QASM-NEXT: brr.anynz r0, r1, -, :vc4_qpu_slot_2 # qpu.branch label=vc4_qpu_slot_0 target=relative:16 target_label=vc4_qpu_slot_2 delay_slots=3
// QASM-NEXT: :vc4_qpu_slot_1
// QASM-NEXT: ldi r0, 0x00000001
// QASM-NEXT: :vc4_qpu_slot_2
// QASM-NEXT: mov ra2, srel2 # sema release 2
// QASM-NEXT: :vc4_qpu_slot_3
// QASM-NEXT: nop
// QASM-NEXT: :vc4_qpu_slot_4
// QASM-NEXT: thrend
// QASM-NEXT: :vc4_qpu_slot_5
// QASM-NEXT: ldi r0, 0x00000000
// QASM-NEXT: :vc4_qpu_slot_6
// QASM-NEXT: nop

// MANIFEST: "kernel": "qpu_branch_kernel"
// MANIFEST: "public_name": "qpu_branch_launch"
// MANIFEST: "scheduled_sink_ops": 7

vc4.module @qpu_branch {
  vc4.func @qpu_branch_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "qpu_branch_launch",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    }
  } {
    vc4.qpu.branch attributes {
      cond = #vc4.branch_cond<any_z_clear>,
      relative = true,
      use_reg = false,
      raddr_a = 0 : i32,
      immediate = 16 : i32,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32
    } {
      vc4.qpu.ldi <splat32> {
        value = 1 : i32,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<never>,
        waddr_add = 32 : i32,
        waddr_mul = 33 : i32
      }

      vc4.qpu.sema <release> {
        id = 2 : i32,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<never>,
        waddr_add = 2 : i32,
        waddr_mul = 3 : i32
      }

      vc4.qpu.bundle {
        sig = #vc4.qpu_signal<none>,
        pm = false,
        cond_add = #vc4.cond<always>,
        cond_mul = #vc4.cond<never>,
        waddr_add = 34 : i32,
        waddr_mul = 35 : i32,
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
      value = 0 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32
    }

    vc4.qpu.bundle {
      sig = #vc4.qpu_signal<none>,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 34 : i32,
      waddr_mul = 35 : i32,
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
