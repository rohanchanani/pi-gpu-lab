// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.func @vertical_dynamic_vdr_row_by_row_kernel
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<read>}
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<read>}
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<read>}
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_addr {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_wait {side = #vc4.vpmvcd_side<read>}
// CHECK-NOT: ssavc4.
ssavc4.module @vertical_dynamic_vdr_row_by_row {
  ssavc4.func @vertical_dynamic_vdr_row_by_row_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vertical_dynamic_vdr_row_by_row",
      code_symbol = "vertical_dynamic_vdr_row_by_row_shader",
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
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %pitch = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    ssavc4.vdr.load_rect.dynamic %in, %row0, %active_rows, %active_cols, %pitch {
      max_rows = 4 : i32,
      max_cols = 8 : i32,
      elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<vertical>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      dst_x = 2 : i32,
      vpm_pitch = 1 : i32,
      zero_fill = true,
      serialize = "mutex"
    } : i32, i32, i32, i32, i32
    ssavc4.thread_end
  }
}
