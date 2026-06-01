// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.func @vdw_store_vpm_dynamic_kernel
// CHECK: op_add = #vc4.add_opcode<max>
// CHECK-SAME: small_imm = 0 : i32
// CHECK: op_add = #vc4.add_opcode<min>
// CHECK-SAME: small_imm = 4 : i32
// CHECK: op_add = #vc4.add_opcode<shl>
// CHECK-SAME: small_imm = 8 : i32
// CHECK: op_add = #vc4.add_opcode<shl>
// CHECK-SAME: small_imm = 8 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-NOT: ssavc4.
ssavc4.module @vdw_store_vpm_dynamic {
  ssavc4.func @vdw_store_vpm_dynamic_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdw_store_vpm_dynamic",
      code_symbol = "vdw_store_vpm_dynamic_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 0 : i32,
      args = [],
      builtins = []
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 1 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 1 : i32,
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
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %y = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %active = ssavc4.load_imm <splat32> {value = 7 : i32} : i32
    ssavc4.vdw.store_vpm %addr, %y, %x, %active {
      width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 4 : i32,
      nrows = 1 : i32,
      memory_pitch_bytes = 16 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      serialize = "mutex"
    } : i32, i32, i32, i32
    ssavc4.thread_end
  }
}
