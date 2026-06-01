// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c

// SOURCE: resident_request_id=resident_block_slot*warps_per_block+logical_warp_id
// SOURCE: spill_frame_base=spill_arena_base+resident_request_id*spill_frame_stride_bytes
// SOURCE-NOT: logical_request*spill_frame_stride
// SOURCE-LABEL: static int spill_cooperative_request_info_pack_uniforms
// SOURCE: uniformWords[0] = requestInfo->spill_frame_base; /* builtin spill_frame_base */
// SOURCE: uniformWords[1] = requestInfo->resident_request_id; /* builtin resident_request_id */
// SOURCE: return vc4LaunchKernel(program, 0u, totalRequests, warpsPerBlock, spill_cooperative_request_info_pack_uniforms, &ctx);

vc4.module @spill_cooperative_request_info {
  vc4.func @kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    spill_frame_bytes = 20 : i32,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "spill_cooperative_request_info",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "spill_frame_base", kind = #vc4.builtin_kind<spill_frame_base>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "resident_request_id", kind = #vc4.builtin_kind<resident_request_id>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 12 : i32,
      user_vpm_rows_per_block = 16 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 16 : i32,
      uses_tmu = false,
      uses_vpm = true,
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = false,
      uses_vdr = false,
      uses_vdw = false,
      uses_barrier = true,
      semaphore_count_per_block = 4 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = true
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
