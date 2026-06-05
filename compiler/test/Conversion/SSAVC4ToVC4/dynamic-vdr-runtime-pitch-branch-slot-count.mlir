// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @dynamic_vdr_runtime_pitch_branch_slot_count
// CHECK-LABEL: vc4.func @kernel
// The first conditional branch skips over the dynamic VDR runtime-pitch body.
// Its immediate must include guarded fallback/planned-region slots.
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_z_set>, immediate = 2400 : i32
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<read>}
// CHECK: vc4.qpu.bundle {{.*}}sig = #vc4.qpu_signal<thrend>
// CHECK-NOT: ssavc4.

ssavc4.module @dynamic_vdr_runtime_pitch_branch_slot_count {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "dynamic_vdr_runtime_pitch_branch_slot_count",
      code_symbol = "dynamic_vdr_runtime_pitch_branch_slot_count_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 6 : i32,
      args = [
        {name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 1 : i32},
        {name = "mode", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
        {name = "active_rows", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
        {name = "active_cols", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32},
        {name = "pitch_bytes", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 5 : i32}
      ],
      builtins = []
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 16 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 16 : i32,
      uses_tmu = false,
      uses_vpm = true,
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = false,
      uses_vdr = true,
      uses_vdw = true,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %in = ssavc4.uniform.read 0 : i32
    %out = ssavc4.uniform.read 1 : i32
    %mode = ssavc4.uniform.read 2 : i32
    %active_rows = ssavc4.uniform.read 3 : i32
    %active_cols = ssavc4.uniform.read 4 : i32
    %pitch = ssavc4.uniform.read 5 : i32
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32

    %flags = ssavc4.make_flags %mode, %zero {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %flags, ^skip_vdr, ^run_vdr {cond = #vc4.branch_cond<any_z_set>} : !ssavc4.flags

  ^run_vdr:
    ssavc4.vdr.load_rect.dynamic %in, %zero, %active_rows, %active_cols, %pitch {
      max_rows = 4 : i32,
      max_cols = 16 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      dst_x = 0 : i32,
      vpm_pitch = 1 : i32,
      zero_fill = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32

    ssavc4.vdw.store_vpm %out, %zero, %zero {
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      row_len = 16 : i32,
      nrows = 4 : i32,
      memory_pitch_bytes = 64 : i32,
      active_lanes = 16 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      serialize = "mutex"
    } : i32, i32, i32
    ssavc4.br ^done

  ^skip_vdr:
    ssavc4.br ^done

  ^done:
    ssavc4.thread_end
  }
}
