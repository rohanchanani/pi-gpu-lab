// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.func @dynamic_vdw_source_row_runtime_kernel
// CHECK: waddr_add = 1 : i32
// CHECK: op_add = #vc4.add_opcode<or>{{.*}}raddr_a = 1 : i32{{.*}}raddr_b = 1 : i32
// CHECK: op_add = #vc4.add_opcode<shl>{{.*}}small_imm = 7 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<write>

// CHECK-LABEL: vc4.func @dynamic_vdw_source_row_constant_kernel
// CHECK: value = 13 : i32
// CHECK: op_add = #vc4.add_opcode<shl>{{.*}}small_imm = 7 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK-NOT: ssavc4.

ssavc4.module @dynamic_vdw_source_row {
  ssavc4.func @dynamic_vdw_source_row_runtime_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "dynamic_vdw_source_row_runtime",
      code_symbol = "dynamic_vdw_source_row_runtime_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
        {name = "source_row", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32}
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
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %stride = ssavc4.load_imm <splat32> {value = 64 : i32} : i32

    ssavc4.vdw.store_rect.dynamic %out, %source_row, %rows, %cols, %stride {
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

  ssavc4.func @dynamic_vdw_source_row_constant_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "dynamic_vdw_source_row_constant",
      code_symbol = "dynamic_vdw_source_row_constant_shader",
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
    %source_row = ssavc4.load_imm <splat32> {value = 13 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %stride = ssavc4.load_imm <splat32> {value = 64 : i32} : i32

    ssavc4.vdw.store_rect.dynamic %out, %source_row, %rows, %cols, %stride {
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
}
