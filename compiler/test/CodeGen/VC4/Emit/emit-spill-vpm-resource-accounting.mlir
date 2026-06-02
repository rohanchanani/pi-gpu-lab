// RUN: rm -rf %t.bundle
// RUN: vc4-codegen %s --emit-bundle %t.bundle
// RUN: FileCheck %s --check-prefix=SOURCE --input-file=%t.bundle/kernel_launch.c
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json

// SOURCE: vpm_base_row=resident_slot*20
// SOURCE: user_vpm_rows=16 compiler_vpm_staging_rows=0 spill_vpm_rows=4
// SOURCE: resident_request_id=resident_block_slot*warps_per_block+logical_warp_id
// SOURCE: spill_vpm_row=vpm_base_row+16+logical_warp_id
// SOURCE: spill_frame_base=spill_arena_base+resident_request_id*spill_frame_stride_bytes
// SOURCE-NOT: logical_request*spill_frame_stride
// SOURCE-LABEL: static int spill_vpm_resource_accounting_pack_uniforms
// SOURCE: uniformWords[0] = requestInfo->spill_frame_base; /* builtin spill_frame_base */
// SOURCE: uniformWords[1] = requestInfo->vpm_base_row + KERNEL_0_USER_VPM_ROWS_PER_BLOCK + KERNEL_0_COMPILER_VPM_STAGING_ROWS_PER_BLOCK + requestInfo->logical_warp_id; /* builtin spill_vpm_row */
// SOURCE: return vc4LaunchKernel(program, 0u, grid, block, totalRequests, warpsPerBlock, spill_vpm_resource_accounting_pack_uniforms, &ctx);
// MANIFEST: "user_vpm_rows_per_block": 16
// MANIFEST: "spill_vpm_rows_per_block": 4
// MANIFEST: "total_vpm_rows_per_block": 20
// MANIFEST: "max_resident_blocks": 3

vc4.module @spill_vpm_resource_accounting {
  vc4.func @kernel() attributes {
    domain = #vc4.execution_domain<qpu>,
    form = #vc4.function_form<scheduled>,
    kernel,
    spill_frame_bytes = 20 : i32,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "spill_vpm_resource_accounting",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "spill_frame_base", kind = #vc4.builtin_kind<spill_frame_base>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "spill_vpm_row", kind = #vc4.builtin_kind<spill_vpm_row>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 4 : i32,
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
