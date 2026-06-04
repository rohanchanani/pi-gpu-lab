// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.func @vdw_zero_cols_prior_dma_kernel
// CHECK: value = 13 : i32
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_c_set>, immediate = 216 : i32
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>

// CHECK-LABEL: vc4.func @vdw_high_rows_dynamic_stride_fallback_kernel
// CHECK: value = 13 : i32
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK: value = 1 : i32
// CHECK: op_add = #vc4.add_opcode<add>
// CHECK: raddr_a = 2 : i32
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK: value = 2 : i32
// CHECK: op_add = #vc4.add_opcode<add>
// CHECK: raddr_a = 2 : i32
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>

// CHECK-LABEL: vc4.func @vdw_static_stride_overflow_fallback_kernel
// CHECK: value = 17000 : i32
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK: value = 1 : i32
// CHECK: op_add = #vc4.add_opcode<add>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK: value = 2 : i32
// CHECK: op_add = #vc4.add_opcode<add>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>

// CHECK-LABEL: vc4.func @vdw_runtime_stride_gap_dispatch_kernel
// CHECK: value = 64 : i32
// CHECK: op_add = #vc4.add_opcode<sub>
// CHECK: value = 8191 : i32
// CHECK: op_add = #vc4.add_opcode<sub>
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_c_set>
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<always>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>

// CHECK-LABEL: vc4.func @vdr_zero_active_shape_kernel
// CHECK: waddr_add = 48 : i32
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_c_set>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>

// CHECK-LABEL: vc4.func @rectangular_fast_paths_retained_kernel
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_wait {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK: vc4.qpu.vpmvcd_wait {{.*}}side = #vc4.vpmvcd_side<write>
// CHECK-NOT: ssavc4.

ssavc4.module @planned_dynamic_dma_invariants {
  ssavc4.func @vdw_zero_cols_prior_dma_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdw_zero_cols_prior_dma",
      code_symbol = "vdw_zero_cols_prior_dma_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
        {name = "active_cols", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32}
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
    %active_cols = ssavc4.uniform.read 1 : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %source_row = ssavc4.load_imm <splat32> {value = 13 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %stride = ssavc4.load_imm <splat32> {value = 64 : i32} : i32

    ssavc4.vdw.store_rect.dynamic %out, %row0, %rows, %cols, %stride {
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

    ssavc4.vdw.store_rect.dynamic %out, %source_row, %rows, %active_cols, %stride {
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

  ssavc4.func @vdw_high_rows_dynamic_stride_fallback_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdw_high_rows_dynamic_stride_fallback",
      code_symbol = "vdw_high_rows_dynamic_stride_fallback_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
        {name = "stride", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32}
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
    %stride = ssavc4.uniform.read 1 : i32
    %source_row = ssavc4.load_imm <splat32> {value = 13 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32

    ssavc4.vdw.store_rect.dynamic %out, %source_row, %rows, %cols, %stride {
      max_rows = 3 : i32,
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

  ssavc4.func @vdw_static_stride_overflow_fallback_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdw_static_stride_overflow_fallback",
      code_symbol = "vdw_static_stride_overflow_fallback_shader",
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
    %source_row = ssavc4.load_imm <splat32> {value = 13 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %stride = ssavc4.load_imm <splat32> {value = 17000 : i32} : i32

    ssavc4.vdw.store_rect.dynamic %out, %source_row, %rows, %cols, %stride {
      max_rows = 3 : i32,
      max_cols = 1 : i32,
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

  ssavc4.func @vdw_runtime_stride_gap_dispatch_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdw_runtime_stride_gap_dispatch",
      code_symbol = "vdw_runtime_stride_gap_dispatch_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 3 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
        {name = "active_rows", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32},
        {name = "stride", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32}
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
    %stride = ssavc4.uniform.read 2 : i32
    %source_row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32

    ssavc4.vdw.store_rect.dynamic %out, %source_row, %active_rows, %cols, %stride {
      max_rows = 2 : i32,
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

  ssavc4.func @vdr_zero_active_shape_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdr_zero_active_shape",
      code_symbol = "vdr_zero_active_shape_shader",
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
      uses_vpm_qpu_write = false,
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

  ssavc4.func @rectangular_fast_paths_retained_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "rectangular_fast_paths_retained",
      code_symbol = "rectangular_fast_paths_retained_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [
        {name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 1 : i32}
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
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %pitch = ssavc4.load_imm <splat32> {value = 64 : i32} : i32

    ssavc4.vdr.load_rect.dynamic %in, %row0, %rows, %cols, %pitch {
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

    ssavc4.vdw.store_rect.dynamic %out, %row0, %rows, %cols, %pitch {
      max_rows = 2 : i32,
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
