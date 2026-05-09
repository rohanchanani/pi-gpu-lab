// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernels/qpu_ldi_sema_launch.qasm
// RUN: test ! -f %t.bundle/kernel.qasm
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=QASM --input-file=%t.bundle/kernels/qpu_ldi_sema_launch.qasm
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json

// QASM: ldi.setf ra0.16a, 0x00001234
// QASM-NEXT: ldi.ifz -, ra9.8a, 0x00000007
// QASM-NEXT: mov ra10, sacq3 # sema acquire 3
// QASM-NEXT: mov.ifnz ra11, srel2 # sema release 2
// QASM-NEXT: thrend
// QASM-NEXT: ldi r0, 0x0000000b
// QASM-NEXT: mov r1, srel1 # sema release 1


// MANIFEST: "schema_version": 2
// MANIFEST: "program_name": "qpu_ldi_sema"
// MANIFEST: "kernels": [
// MANIFEST: "kernel_id": 0
// MANIFEST: "symbol_name": "qpu_ldi_sema_kernel"
// MANIFEST: "public_name": "qpu_ldi_sema_launch"
// MANIFEST: "qasm_path": "kernels/qpu_ldi_sema_launch.qasm"
// MANIFEST: "code_symbol": "qpu_ldi_sema_launch_shader"
// MANIFEST: "scheduled_sink_ops": 7
vc4.module @qpu_ldi_sema {
  vc4.func @qpu_ldi_sema_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "qpu_ldi_sema_launch",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    }
  } {
    vc4.qpu.ldi <splat32> {
      value = 4660 : i32,
      pm = false,
      pack = #vc4.regfile_a_pack_mode<to_16a>,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      set_flags,
      waddr_add = 0 : i32,
      waddr_mul = 1 : i32
    }

    vc4.qpu.ldi <splat32> {
      value = 7 : i32,
      pm = true,
      pack = #vc4.mul_pack_mode<to_8a>,
      cond_add = #vc4.cond<never>,
      cond_mul = #vc4.cond<zs>,
      write_swap,
      waddr_add = 8 : i32,
      waddr_mul = 9 : i32
    }

    vc4.qpu.sema <acquire> {
      id = 3 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 10 : i32,
      waddr_mul = 11 : i32
    }

    vc4.qpu.sema <release> {
      id = 2 : i32,
      pm = false,
      cond_add = #vc4.cond<zc>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 11 : i32,
      waddr_mul = 12 : i32
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
      waddr_add = 32 : i32,
      waddr_mul = 33 : i32
    }

    vc4.qpu.sema <release> {
      id = 1 : i32,
      pm = false,
      cond_add = #vc4.cond<always>,
      cond_mul = #vc4.cond<never>,
      waddr_add = 33 : i32,
      waddr_mul = 34 : i32
    }
  }
}
