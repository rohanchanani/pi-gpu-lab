// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernels/uniform_packing.qasm
// RUN: test ! -f %t.bundle/kernel.qasm
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=HEADER --input-file=%t.bundle/kernel_launch.h
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json

// HEADER: int uniform_packing_launch(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block, float scale, vc4_deviceptr_t out, uint32_t n, vc4_deviceptr_t input);
// HEADER-NOT: struct vc4_runtime
// HEADER-NOT: uniform_packing_prepare
// HEADER-NOT: uniform_packing_release
// HEADER-NOT: uint32_t *out
// HEADER-NOT: const float *input
// HEADER-NOT: uint32_t *uniform
// HEADER-NOT: qpu_id
// HEADER-NOT: num_qpus

// SOURCE: static uint32_t vc4_codegen_pack_f32(float value) {
// SOURCE-LABEL: int uniform_packing_launch(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block, float scale, vc4_deviceptr_t out, uint32_t n, vc4_deviceptr_t input) {
// SOURCE-NOT: vc4Malloc(
// SOURCE-NOT: vc4MemcpyHtoD(
// SOURCE-NOT: vc4MemcpyDtoH(
// SOURCE-NOT: vc4Free(
// SOURCE: uint32_t activeQpus
// SOURCE: * Dense physical uniform layout per logical request, ordered by vc4.launch_abi uniform_index:
// SOURCE: *   [0] arg out
// SOURCE: *   [1] arg input
// SOURCE: *   [2] arg n
// SOURCE: *   [3] arg scale
// SOURCE: *   [4] builtin qpu_id (qpu_num)
// SOURCE: *   [5] builtin num_qpus (num_qpus)
// SOURCE: for (uint32_t qpu = 0; qpu < waveRequests; ++qpu) {
// SOURCE: [qpu][0] = (uint32_t)out; /* arg out */
// SOURCE: [qpu][1] = (uint32_t)input; /* arg input */
// SOURCE: [qpu][2] = (uint32_t)n; /* arg n */
// SOURCE: [qpu][3] = vc4_codegen_pack_f32(scale); /* arg scale */
// SOURCE: [qpu][4] = logicalRequest; /* builtin qpu_id */
// SOURCE: [qpu][5] = totalRequests; /* builtin num_qpus */
// SOURCE: PUT32(V3D_SRQUA,
// SOURCE: PUT32(V3D_SRQPC,
// SOURCE: return 0;

// MANIFEST: "schema_version": 2
// MANIFEST: "program_name": "uniform_packing"
// MANIFEST: "kernels": [
// MANIFEST: "kernel_id": 0
// MANIFEST: "symbol_name": "uniform_packing_kernel"
// MANIFEST: "public_name": "uniform_packing"
// MANIFEST: "qasm_path": "kernels/uniform_packing.qasm"
// MANIFEST: "code_symbol": "uniform_packing_shader"
// MANIFEST-DAG: {"name": "scale", "kind": "scalar", "direction": "by_value", "type": "f32", "c_type": "float", "uniform_index": 3}
// MANIFEST-DAG: {"name": "out", "kind": "buffer", "direction": "out", "elem_type": "u32", "c_type": "vc4_deviceptr_t", "uniform_index": 0}
// MANIFEST-DAG: {"name": "n", "kind": "scalar", "direction": "by_value", "type": "u32", "c_type": "uint32_t", "uniform_index": 2}
// MANIFEST-DAG: {"name": "input", "kind": "buffer", "direction": "in", "elem_type": "f32", "c_type": "vc4_deviceptr_t", "uniform_index": 1}

vc4.module @uniform_packing {
  vc4.func @uniform_packing_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "uniform_packing",
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
