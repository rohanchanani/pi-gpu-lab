// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.func @dynamic_rows_static_partial_cols_kernel
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK-NOT: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK-LABEL: vc4.func @dynamic_rows_cols_kernel
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK-NOT: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK-LABEL: vc4.func @zero_rows_skip_vdr_kernel
// CHECK-NOT: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK-LABEL: vc4.func @zero_cols_skip_vdr_kernel
// CHECK-NOT: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK-LABEL: vc4.func @runtime_pitch_overflow_fallback_kernel
// CHECK: value = 8191 : i32
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<always>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK-NOT: ssavc4.
ssavc4.module @vdr_true_rect_dynamic_shape {
  ssavc4.func @dynamic_rows_static_partial_cols_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "dynamic_rows_static_partial_cols",
      code_symbol = "dynamic_rows_static_partial_cols_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [
        {name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "active_rows", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32}
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
      uses_vpm_qpu_write = true,
      uses_vdr = true,
      uses_vdw = false,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %in = ssavc4.uniform.read 0 : i32
    %active_rows = ssavc4.uniform.read 1 : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %cols5 = ssavc4.load_imm <splat32> {value = 5 : i32} : i32
    %pitch = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    ssavc4.vdr.load_rect.dynamic %in, %row0, %active_rows, %cols5, %pitch {
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
    ssavc4.thread_end
  }

  ssavc4.func @dynamic_rows_cols_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "dynamic_rows_cols",
      code_symbol = "dynamic_rows_cols_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 3 : i32,
      args = [
        {name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "active_rows", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32},
        {name = "active_cols", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32}
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
      uses_vpm_qpu_write = true,
      uses_vdr = true,
      uses_vdw = false,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %in = ssavc4.uniform.read 0 : i32
    %active_rows = ssavc4.uniform.read 1 : i32
    %active_cols = ssavc4.uniform.read 2 : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %pitch = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    ssavc4.vdr.load_rect.dynamic %in, %row0, %active_rows, %active_cols, %pitch {
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
    ssavc4.thread_end
  }

  ssavc4.func @zero_rows_skip_vdr_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "zero_rows_skip_vdr",
      code_symbol = "zero_rows_skip_vdr_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 1 : i32,
      args = [{name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32}],
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
      uses_vpm_qpu_write = true,
      uses_vdr = true,
      uses_vdw = false,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %in = ssavc4.uniform.read 0 : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %rows0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %cols5 = ssavc4.load_imm <splat32> {value = 5 : i32} : i32
    %pitch = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    ssavc4.vdr.load_rect.dynamic %in, %row0, %rows0, %cols5, %pitch {
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
    ssavc4.thread_end
  }

  ssavc4.func @zero_cols_skip_vdr_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "zero_cols_skip_vdr",
      code_symbol = "zero_cols_skip_vdr_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 1 : i32,
      args = [{name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32}],
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
      uses_vpm_qpu_write = true,
      uses_vdr = true,
      uses_vdw = false,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %in = ssavc4.uniform.read 0 : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %rows4 = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %cols0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %pitch = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    ssavc4.vdr.load_rect.dynamic %in, %row0, %rows4, %cols0, %pitch {
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
    ssavc4.thread_end
  }

  ssavc4.func @runtime_pitch_overflow_fallback_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "runtime_pitch_overflow_fallback",
      code_symbol = "runtime_pitch_overflow_fallback_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 4 : i32,
      args = [
        {name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "active_rows", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32},
        {name = "active_cols", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
        {name = "pitch", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32}
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
      uses_vpm_qpu_write = true,
      uses_vdr = true,
      uses_vdw = false,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %in = ssavc4.uniform.read 0 : i32
    %active_rows = ssavc4.uniform.read 1 : i32
    %active_cols = ssavc4.uniform.read 2 : i32
    %pitch = ssavc4.uniform.read 3 : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    ssavc4.vdr.load_rect.dynamic %in, %row0, %active_rows, %active_cols, %pitch {
      max_rows = 2 : i32,
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
    ssavc4.thread_end
  }
}
