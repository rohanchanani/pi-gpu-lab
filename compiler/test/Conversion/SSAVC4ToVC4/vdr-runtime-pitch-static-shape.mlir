// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.func @vdr_static_16x16_runtime_pitch_kernel
// CHECK: value = 8191 : i32
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_c_set>
// CHECK: value = -1879048192 : i32
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}raddr_a = 1 : i32{{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: value = -2147479552 : i32
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}op_add = #vc4.add_opcode<or>{{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<always>

// CHECK-LABEL: vc4.func @vdr_static_7x5_runtime_pitch_kernel
// CHECK: waddr_add = 48 : i32
// CHECK: waddr_add = 48 : i32
// CHECK: value = 8191 : i32
// CHECK: value = -1879048192 : i32
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}raddr_a = 1 : i32{{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}op_add = #vc4.add_opcode<or>{{.*}}side = #vc4.vpmvcd_side<read>
// CHECK-NOT: ssavc4.

ssavc4.module @vdr_runtime_pitch_static_shape {
  ssavc4.func @vdr_static_16x16_runtime_pitch_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdr_static_16x16_runtime_pitch",
      code_symbol = "vdr_static_16x16_runtime_pitch_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [
        {name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "pitch", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32}
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
    %pitch = ssavc4.uniform.read 1 : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32

    ssavc4.vdr.load_rect.dynamic %in, %row0, %rows, %cols, %pitch {
      max_rows = 16 : i32,
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

  ssavc4.func @vdr_static_7x5_runtime_pitch_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdr_static_7x5_runtime_pitch",
      code_symbol = "vdr_static_7x5_runtime_pitch_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [
        {name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "pitch", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32}
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
    %pitch = ssavc4.uniform.read 1 : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 7 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 5 : i32} : i32

    ssavc4.vdr.load_rect.dynamic %in, %row0, %rows, %cols, %pitch {
      max_rows = 16 : i32,
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
