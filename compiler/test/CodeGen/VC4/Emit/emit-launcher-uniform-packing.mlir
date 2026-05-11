// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernels/uniform_packing.qasm
// RUN: test ! -f %t.bundle/kernel.qasm
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=HEADER --input-file=%t.bundle/kernel_launch.h --implicit-check-not='"mailbox.h"' --implicit-check-not='typedef uint32_t vc4_deviceptr_t'
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c --implicit-check-not='struct vc4_program {' --implicit-check-not='struct vc4_codegen_heap_block' --implicit-check-not='int vc4Malloc' --implicit-check-not='int vc4Memcpy' --implicit-check-not='qpu_enable(' --implicit-check-not='mem_alloc(' --implicit-check-not='mem_lock(' --implicit-check-not='mem_free(' --implicit-check-not='gpu_fft_base_exec_direct' --implicit-check-not='V3D_' --implicit-check-not='PUT32(' --implicit-check-not='GET32(' --implicit-check-not='VC4_HEAP_STATS'
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json

// HEADER: #include <stdint.h>
// HEADER: #include "vc4_runtime.h"
// HEADER: int vc4_program_create(struct vc4_program **out, uint32_t requested_bytes);
// HEADER: int uniform_packing_launch(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block, float scale, vc4_deviceptr_t out, uint32_t n, vc4_deviceptr_t input);
// HEADER-NOT: struct vc4_runtime
// HEADER-NOT: uniform_packing_prepare
// HEADER-NOT: uniform_packing_release
// HEADER-NOT: uint32_t *out
// HEADER-NOT: const float *input
// HEADER-NOT: uint32_t *uniform
// HEADER-NOT: qpu_id
// HEADER-NOT: num_qpus

// SOURCE: #include "kernel_launch.h"
// SOURCE: static uint32_t vc4_codegen_pack_f32(float value) {
// SOURCE: static const struct vc4_kernel_image vc4_codegen_kernels[] = {
// SOURCE: { "uniform_packing", uniform_packing_shader,
// SOURCE: VC4_KERNEL_SCHEDULE_INDEPENDENT_VECTOR
// SOURCE: static const struct vc4_module_image vc4_codegen_module = {
// SOURCE: int vc4_program_create(struct vc4_program **out, uint32_t requested_bytes) {
// SOURCE: return vc4ProgramCreateFromImage(out, &vc4_codegen_module, requested_bytes);
// SOURCE: struct uniform_packing_pack_ctx {
// SOURCE-LABEL: static int uniform_packing_pack_uniforms(void *opaque, uint32_t logicalRequest, uint32_t *uniformWords, uint32_t uniformWordsPerRequest) {
// SOURCE: * Dense physical uniform layout per logical request, ordered by vc4.launch_abi uniform_index:
// SOURCE: *   [0] arg out
// SOURCE: *   [1] arg input
// SOURCE: *   [2] arg n
// SOURCE: *   [3] arg scale
// SOURCE: *   [4] builtin qpu_id (qpu_num)
// SOURCE: *   [5] builtin num_qpus (num_qpus)
// SOURCE: uniformWords[0] = (uint32_t)ctx->out; /* arg out */
// SOURCE: uniformWords[1] = (uint32_t)ctx->input; /* arg input */
// SOURCE: uniformWords[2] = (uint32_t)ctx->n; /* arg n */
// SOURCE: uniformWords[3] = vc4_codegen_pack_f32(ctx->scale); /* arg scale */
// SOURCE: uniformWords[4] = logicalRequest; /* builtin qpu_id */
// SOURCE: uniformWords[5] = ctx->total_requests; /* builtin num_qpus */
// SOURCE-LABEL: int uniform_packing_launch(struct vc4_program *program, vc4_dim3 grid, vc4_dim3 block, float scale, vc4_deviceptr_t out, uint32_t n, vc4_deviceptr_t input) {
// SOURCE: vc4DeviceRangeIsAllocated(program, out,
// SOURCE: vc4DeviceRangeIsAllocated(program, input,
// SOURCE: return vc4LaunchKernel(program, 0u, totalRequests, uniform_packing_pack_uniforms, &ctx);

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
