// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c --implicit-check-not='vc4Malloc(' --implicit-check-not='vc4Free(' --implicit-check-not='vc4Memcpy'
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=LAYOUT --input-file=%t.bundle/layout.json

// SOURCE: #define VC4_CODEGEN_PROGRAM_HEAP_BYTES 65536u
// SOURCE: #define VC4_CODEGEN_SPILL_ARENA_BYTES 0u
// SOURCE: #define KERNEL_0_SPILL_FRAME_BYTES 0u
// SOURCE: #define KERNEL_0_SPILL_FRAME_STRIDE_BYTES 0u
// SOURCE: #define KERNEL_0_SPILL_FRAME_COUNT 0u
// SOURCE: #define KERNEL_0_SPILL_ARENA_BYTES 0u
// SOURCE: hidden_spill_arena_bytes=0
// SOURCE: hidden_arena_before_public_heap=0

// MANIFEST: "max_spill_arena_bytes": 0
// MANIFEST: "spill_frame_bytes": 0
// MANIFEST: "spill_frame_stride_bytes": 0
// MANIFEST: "spill_frame_count": 0
// MANIFEST: "spill_arena_bytes": 0

// LAYOUT: "hidden_spill_arena_bytes": 0
// LAYOUT: "heap_bytes": 65536
// LAYOUT: "name": "heap"
// LAYOUT-SAME: "kind": "heap"

vc4.module @spill_zero_default {
  vc4.func @kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "spill_zero_default",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 1 : i32}
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
    vc4.qpu.ldi <splat32> {value = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.ldi <splat32> {value = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
  }
}
