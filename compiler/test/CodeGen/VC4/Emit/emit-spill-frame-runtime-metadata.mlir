// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c --implicit-check-not='vc4Malloc(' --implicit-check-not='vc4Free(' --implicit-check-not='vc4Memcpy'
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=LAYOUT --input-file=%t.bundle/layout.json

// SOURCE: #define VC4_CODEGEN_SPILL_FRAME_ALIGNMENT 64u
// SOURCE: #define VC4_CODEGEN_SPILL_ARENA_BYTES 768u
// SOURCE: #define KERNEL_0_SPILL_FRAME_BYTES 20u
// SOURCE: #define KERNEL_0_SPILL_FRAME_STRIDE_BYTES 64u
// SOURCE: #define KERNEL_0_SPILL_FRAME_COUNT 12u
// SOURCE: #define KERNEL_0_SPILL_ARENA_BYTES 768u
// SOURCE: hidden_spill_arena_bytes=768
// SOURCE: hidden_arena_before_public_heap=1
// SOURCE: spill_arena_bytes=768
// SOURCE: spill_frame_base=spill_arena_base+resident_request_id*spill_frame_stride_bytes

// MANIFEST: "max_spill_arena_bytes": 768
// MANIFEST: "spill_frame_bytes": 20
// MANIFEST: "spill_frame_stride_bytes": 64
// MANIFEST: "spill_frame_count": 12
// MANIFEST: "spill_arena_bytes": 768

// LAYOUT: "max_spill_arena_bytes": 768
// LAYOUT: "hidden_spill_arena_bytes": 768
// LAYOUT: "name": "hidden_spill_arena"
// LAYOUT-SAME: "kind": "hidden_spill_arena"
// LAYOUT: "name": "heap"
// LAYOUT-SAME: "kind": "heap"

vc4.module @spill_runtime_metadata {
  vc4.func @kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    spill_frame_bytes = 20 : i32,
    spill_frame_stride_bytes = 64 : i32,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "spill_runtime_metadata",
      tail_policy = "tail_safe",
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
    vc4.qpu.ldi <splat32> {value = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
    vc4.qpu.ldi <splat32> {value = 0 : i32, pm = false, cond_add = #vc4.cond<always>, cond_mul = #vc4.cond<never>, waddr_add = 32 : i32, waddr_mul = 33 : i32}
  }
}
