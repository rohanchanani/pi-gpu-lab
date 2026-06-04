// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.func @vdw_dynamic_rows_static_partial_cols_kernel
// CHECK: value = 5 : i32
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK-NOT: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>

// CHECK-LABEL: vc4.func @vdw_dynamic_rows_cols_source_row_kernel
// CHECK: value = 16 : i32
// CHECK: small_imm = 7 : i32
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK-NOT: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>

// CHECK-LABEL: vc4.func @vdw_zero_cols_after_prior_dma_kernel
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK-NOT: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>

// CHECK-LABEL: vc4.func @vdw_stride_gap_overflow_fallback_kernel
// CHECK: value = 8191 : i32
// CHECK: vc4.qpu.branch
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK: op_add = #vc4.add_opcode<add>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>

// CHECK-LABEL: vc4.func @vdw_count16_encoding_kernel
// CHECK: value = 16 : i32
// CHECK: small_imm = 7 : i32
// CHECK-NOT: small_imm = 15 : i32
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK-NOT: ssavc4.

ssavc4.module @vdw_true_rect_dynamic_shape {
  ssavc4.func @vdw_dynamic_rows_static_partial_cols_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdw_dynamic_rows_static_partial_cols",
      code_symbol = "vdw_dynamic_rows_static_partial_cols_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 3 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
        {name = "source_row", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32},
        {name = "active_rows", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32}
      ],
      builtins = []
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 64 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 64 : i32,
      uses_tmu = false,
      uses_vpm = true,
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = false,
      uses_vdr = false,
      uses_vdw = true,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %out = ssavc4.uniform.read 0 : i32
    %source_row = ssavc4.uniform.read 1 : i32
    %active_rows = ssavc4.uniform.read 2 : i32
    %cols5 = ssavc4.load_imm <splat32> {value = 5 : i32} : i32
    %stride = ssavc4.load_imm <splat32> {value = 68 : i32} : i32

    ssavc4.vdw.store_rect.dynamic %out, %source_row, %active_rows, %cols5, %stride {
      max_rows = 16 : i32,
      max_cols = 16 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      src_x = 0 : i32,
      vpm_pitch = 1 : i32,
      preserve_inactive = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32

    ssavc4.thread_end
  }

  ssavc4.func @vdw_dynamic_rows_cols_source_row_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdw_dynamic_rows_cols_source_row",
      code_symbol = "vdw_dynamic_rows_cols_source_row_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 4 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
        {name = "source_row", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32},
        {name = "active_rows", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
        {name = "active_cols", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32}
      ],
      builtins = []
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 64 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 64 : i32,
      uses_tmu = false,
      uses_vpm = true,
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = false,
      uses_vdr = false,
      uses_vdw = true,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %out = ssavc4.uniform.read 0 : i32
    %source_row = ssavc4.uniform.read 1 : i32
    %active_rows = ssavc4.uniform.read 2 : i32
    %active_cols = ssavc4.uniform.read 3 : i32
    %stride = ssavc4.load_imm <splat32> {value = 68 : i32} : i32

    ssavc4.vdw.store_rect.dynamic %out, %source_row, %active_rows, %active_cols, %stride {
      max_rows = 16 : i32,
      max_cols = 16 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      src_x = 0 : i32,
      vpm_pitch = 1 : i32,
      preserve_inactive = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32

    ssavc4.thread_end
  }

  ssavc4.func @vdw_zero_cols_after_prior_dma_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdw_zero_cols_after_prior_dma",
      code_symbol = "vdw_zero_cols_after_prior_dma_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 1 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32}
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
      uses_vdr = false,
      uses_vdw = true,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %out = ssavc4.uniform.read 0 : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %rows1 = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols16 = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %cols0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %stride = ssavc4.load_imm <splat32> {value = 64 : i32} : i32

    ssavc4.vdw.store_rect.dynamic %out, %row0, %rows1, %cols16, %stride {
      max_rows = 1 : i32,
      max_cols = 16 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      src_x = 0 : i32,
      vpm_pitch = 1 : i32,
      preserve_inactive = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32

    ssavc4.vdw.store_rect.dynamic %out, %row0, %rows1, %cols0, %stride {
      max_rows = 1 : i32,
      max_cols = 16 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      src_x = 0 : i32,
      vpm_pitch = 1 : i32,
      preserve_inactive = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32

    ssavc4.thread_end
  }

  ssavc4.func @vdw_stride_gap_overflow_fallback_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdw_stride_gap_overflow_fallback",
      code_symbol = "vdw_stride_gap_overflow_fallback_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 3 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
        {name = "active_rows", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32},
        {name = "active_cols", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32}
      ],
      builtins = []
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 64 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 64 : i32,
      uses_tmu = false,
      uses_vpm = true,
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = false,
      uses_vdr = false,
      uses_vdw = true,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %out = ssavc4.uniform.read 0 : i32
    %active_rows = ssavc4.uniform.read 1 : i32
    %active_cols = ssavc4.uniform.read 2 : i32
    %source_row = ssavc4.load_imm <splat32> {value = 13 : i32} : i32
    %stride = ssavc4.load_imm <splat32> {value = 17000 : i32} : i32

    ssavc4.vdw.store_rect.dynamic %out, %source_row, %active_rows, %active_cols, %stride {
      max_rows = 16 : i32,
      max_cols = 16 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      src_x = 0 : i32,
      vpm_pitch = 1 : i32,
      preserve_inactive = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32

    ssavc4.thread_end
  }

  ssavc4.func @vdw_count16_encoding_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdw_count16_encoding",
      code_symbol = "vdw_count16_encoding_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
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
      uses_vpm_qpu_write = false,
      uses_vdr = false,
      uses_vdw = true,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %out = ssavc4.uniform.read 0 : i32
    %active_rows = ssavc4.uniform.read 1 : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %cols16 = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %stride = ssavc4.load_imm <splat32> {value = 64 : i32} : i32

    ssavc4.vdw.store_rect.dynamic %out, %row0, %active_rows, %cols16, %stride {
      max_rows = 16 : i32,
      max_cols = 16 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      src_x = 0 : i32,
      vpm_pitch = 1 : i32,
      preserve_inactive = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32

    ssavc4.thread_end
  }
}
