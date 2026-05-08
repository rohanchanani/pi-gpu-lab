// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernel.qasm
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=HEADER --input-file=%t.bundle/kernel_launch.h
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json

// HEADER: int uniform_packing_launch(struct vc4_runtime *rt, float scale, uint32_t *out, uint32_t n, const float *input);
// HEADER-NOT: qpu_id
// HEADER-NOT: num_qpus
// HEADER-NOT: uint32_t *uniform

// SOURCE: #define NUM_UNIFS 6u
// SOURCE: static uint32_t vc4_codegen_pack_f32(float value) {
// SOURCE: struct uniform_packing_launch_state {
// SOURCE: uint32_t code[sizeof(kernelshader) / sizeof(uint32_t)];
// SOURCE: uint32_t unif[VC4_RUNTIME_MAX_QPUS][NUM_UNIFS];
// SOURCE: uint32_t unif_ptr[VC4_RUNTIME_MAX_QPUS];
// SOURCE: int uniform_packing_launch(struct vc4_runtime *rt, float scale, uint32_t *out, uint32_t n, const float *input) {
// SOURCE: uint32_t activeQpus = vc4_runtime_active_qpus(rt);
// SOURCE: * Dense physical uniform layout per QPU, ordered by vc4.launch_abi uniform_index:
// SOURCE: *   [0] arg out
// SOURCE: *   [1] arg input
// SOURCE: *   [2] arg n
// SOURCE: *   [3] arg scale
// SOURCE: *   [4] builtin qpu_id (qpu_num)
// SOURCE: *   [5] builtin num_qpus (num_qpus)
// SOURCE: for (uint32_t qpu = 0; qpu < activeQpus; ++qpu) {
// SOURCE: g_state->unif[qpu][0] = GPU_BASE + (uint32_t)gpu_out; /* arg out */
// SOURCE: g_state->unif[qpu][1] = GPU_BASE + (uint32_t)gpu_input; /* arg input */
// SOURCE: g_state->unif[qpu][2] = (uint32_t)n; /* arg n */
// SOURCE: g_state->unif[qpu][3] = vc4_codegen_pack_f32(scale); /* arg scale */
// SOURCE: g_state->unif[qpu][4] = qpu; /* builtin qpu_id */
// SOURCE: g_state->unif[qpu][5] = activeQpus; /* builtin num_qpus */
// SOURCE: g_state->unif_ptr[qpu] = GPU_BASE + (uint32_t)&g_state->unif[qpu][0];

// MANIFEST: "kernel": "uniform_packing_kernel"
// MANIFEST: "public_name": "uniform_packing_launch"
// MANIFEST: "arg_count": 4

vc4.module @uniform_packing {
  vc4.func @uniform_packing_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "uniform_packing_launch",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 6 : i32,
      args = [
        {name = "scale", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 3 : i32},
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
        {name = "input", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 1 : i32}
      ],
      builtins = [
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 5 : i32},
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 4 : i32}
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
