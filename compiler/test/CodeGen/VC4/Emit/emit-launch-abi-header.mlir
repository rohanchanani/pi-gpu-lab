// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernels/launch_abi_header_launch.qasm
// RUN: test ! -f %t.bundle/kernel.qasm
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=HEADER --input-file=%t.bundle/kernel_launch.h
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json

// HEADER: #include <stdint.h>
// HEADER: #include "mailbox.h"
// HEADER-DAG: typedef uint32_t vc4_deviceptr_t;
// HEADER-DAG: typedef struct vc4_dim3
// HEADER-DAG: struct vc4_program;
// HEADER-DAG: int vc4_program_create(struct vc4_program **out, uint32_t requested_bytes);
// HEADER-DAG: int vc4Malloc(struct vc4_program *program, vc4_deviceptr_t *out, uint32_t bytes);
// HEADER-DAG: int vc4MemcpyHtoD(struct vc4_program *program, vc4_deviceptr_t dst, const void *src, uint32_t bytes);
// HEADER-DAG: int vc4MemcpyDtoH(struct vc4_program *program, void *dst, vc4_deviceptr_t src, uint32_t bytes);
// HEADER-DAG: int vc4Free(struct vc4_program *program, vc4_deviceptr_t ptr);
// HEADER-DAG: int launch_abi_header_launch(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block, vc4_deviceptr_t out, vc4_deviceptr_t input, uint32_t n, float scale);
// HEADER-NOT: struct vc4_runtime
// HEADER-NOT: launch_abi_header_prepare
// HEADER-NOT: launch_abi_header_release
// HEADER-NOT: uint32_t *out
// HEADER-NOT: const float *input
// HEADER-NOT: uint32_t *uniform
// HEADER-NOT: qpu_id
// HEADER-NOT: num_qpus

// SOURCE: #include "kernel_launch.h"
// SOURCE: #define VC4_CODEGEN_QPU_WAIT_MAX_POLLS 10000000u
// SOURCE: static uint32_t vc4_codegen_pack_f32(float value) {
// SOURCE: static int vc4_codegen_wait_for_qpus(struct vc4_program *program, uint32_t activeQpus) {
// SOURCE: vc4_codegen_launch_failure(program);
// SOURCE: return -1;
// SOURCE-LABEL: int launch_abi_header_launch(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block, vc4_deviceptr_t out, vc4_deviceptr_t input, uint32_t n, float scale) {
// SOURCE-NOT: vc4Malloc(
// SOURCE-NOT: vc4MemcpyHtoD(
// SOURCE-NOT: vc4MemcpyDtoH(
// SOURCE-NOT: vc4Free(
// SOURCE: uint32_t activeQpus
// SOURCE: * Dense physical uniform layout per QPU, ordered by vc4.launch_abi uniform_index:
// SOURCE: *   [0] arg out
// SOURCE: *   [1] arg input
// SOURCE: *   [2] arg n
// SOURCE: *   [3] arg scale
// SOURCE: *   [4] builtin qpu_id (qpu_num)
// SOURCE: *   [5] builtin num_qpus (num_qpus)
// SOURCE: for (uint32_t qpu = 0; qpu < activeQpus; ++qpu) {
// SOURCE: [qpu][0] = (uint32_t)out; /* arg out */
// SOURCE: [qpu][1] = (uint32_t)input; /* arg input */
// SOURCE: [qpu][2] = (uint32_t)n; /* arg n */
// SOURCE: [qpu][3] = vc4_codegen_pack_f32(scale); /* arg scale */
// SOURCE: [qpu][4] = qpu; /* builtin qpu_id */
// SOURCE: [qpu][5] = activeQpus; /* builtin num_qpus */
// SOURCE: VC4_KERNEL_LAUNCH name=launch_abi_header_launch
// SOURCE: uint32_t max_wait_polls = VC4_CODEGEN_QPU_WAIT_MAX_POLLS;
// SOURCE: PUT32(V3D_SRQUA,
// SOURCE: PUT32(V3D_SRQPC,
// SOURCE: vc4_codegen_launch_code_gpu_addr(program->state->kernel_descs[0].code_gpu_addr)
// SOURCE: if (vc4_codegen_wait_for_qpus(program, activeQpus) < 0)
// SOURCE: return -1;
// SOURCE: program->state->launch_count++;
// SOURCE: return 0;

// MANIFEST: "schema_version": 2
// MANIFEST: "program_name": "launch_abi_header"
// MANIFEST: "kernels": [
// MANIFEST: "kernel_id": 0
// MANIFEST: "symbol_name": "launch_abi_header_kernel"
// MANIFEST: "public_name": "launch_abi_header_launch"
// MANIFEST: "qasm_path": "kernels/launch_abi_header_launch.qasm"
// MANIFEST: "code_symbol": "launch_abi_header_launch_shader"
// MANIFEST-DAG: {"name": "out", "kind": "buffer", "direction": "out", "elem_type": "u32", "c_type": "vc4_deviceptr_t", "uniform_index": 0}
// MANIFEST-DAG: {"name": "input", "kind": "buffer", "direction": "in", "elem_type": "f32", "c_type": "vc4_deviceptr_t", "uniform_index": 1}
// MANIFEST-DAG: {"name": "n", "kind": "scalar", "direction": "by_value", "type": "u32", "c_type": "uint32_t", "uniform_index": 2}
// MANIFEST-DAG: {"name": "scale", "kind": "scalar", "direction": "by_value", "type": "f32", "c_type": "float", "uniform_index": 3}
vc4.module @launch_abi_header {
  vc4.func @launch_abi_header_kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "launch_abi_header_launch",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 6 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
        {name = "input", kind = "buffer", direction = "in", elem_type = "f32", uniform_index = 1 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
        {name = "scale", kind = "scalar", direction = "by_value", type = "f32", uniform_index = 3 : i32}
      ],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 4 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 5 : i32}
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
