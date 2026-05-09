// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: python3 %S/../Support/check_m2_manifest.py %t.bundle/manifest.json --schema-version 2 --kernel-count 1 --require-kernel-fields --arg-direction src:in --arg-direction dst:out --arg-direction scratch:inout
// RUN: FileCheck %s --check-prefix=HEADER --input-file=%t.bundle/kernel_launch.h

// Buffer directions are metadata for user/runtime tooling. The generated
// launch function packs uniforms and enqueues; it must not be a host-copy
// convenience wrapper.
// HEADER-DAG: int buffer_direction_launch(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block, vc4_deviceptr_t src, vc4_deviceptr_t dst, vc4_deviceptr_t scratch, uint32_t n);
// HEADER-NOT: uint32_t *src
// HEADER-NOT: uint32_t *dst
// HEADER-NOT: uint32_t *scratch

vc4.module @buffer_direction_metadata {

  vc4.func @buffer_direction_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "buffer_direction",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 6 : i32,
      args = [
        {name = "src", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "dst", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 1 : i32},
        {name = "scratch", kind = "buffer", direction = "inout", elem_type = "u32", uniform_index = 2 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32}
      ],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 4 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 5 : i32}
      ]
    }
  } {
    // Thread end plus two non-branch scheduled delay-slot instructions.
    // The two trailing bundles are hardware nops: both ALU pipes are inactive.
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
