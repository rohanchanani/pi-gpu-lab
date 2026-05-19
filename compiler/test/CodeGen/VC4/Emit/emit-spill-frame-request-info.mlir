// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c

// SOURCE: resident_request_id=wave_local_request_index
// SOURCE: spill_frame_base=spill_arena_base+resident_request_id*spill_frame_stride_bytes
// SOURCE-LABEL: static int spill_request_info_pack_uniforms
// SOURCE: uniformWords[0] = requestInfo->spill_frame_base; /* hidden_runtime __vc4_spill_frame_base */
// SOURCE: uniformWords[1] = requestInfo->resident_request_id; /* hidden_runtime __vc4_resident_request_id */
// SOURCE: uniformWords[2] = requestInfo->spill_frame_bytes; /* hidden_runtime __vc4_spill_frame_bytes */
// SOURCE: uniformWords[3] = requestInfo->spill_frame_stride_bytes; /* hidden_runtime __vc4_spill_frame_stride_bytes */

vc4.module @spill_request_info {
  vc4.func @kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    spill_frame_bytes = 20 : i32,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "spill_request_info",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 4 : i32,
      args = [],
      builtins = [
        {name = "__vc4_spill_frame_base", kind = "hidden_runtime", materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "__vc4_resident_request_id", kind = "hidden_runtime", materialization = "uniform_suffix", uniform_index = 1 : i32},
        {name = "__vc4_spill_frame_bytes", kind = "hidden_runtime", materialization = "uniform_suffix", uniform_index = 2 : i32},
        {name = "__vc4_spill_frame_stride_bytes", kind = "hidden_runtime", materialization = "uniform_suffix", uniform_index = 3 : i32}
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
