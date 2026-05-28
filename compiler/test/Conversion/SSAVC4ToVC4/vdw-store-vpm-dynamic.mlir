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
      uses_barrier = false,
      uses_shared_vpm = true,
      shared_vpm_bytes = 64 : i32,
      require_full_block_residency = true,
      semaphores_per_block = 4 : i32,
      warps_per_block_max = 1 : i32
    }
  } {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %y = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %active = ssavc4.load_imm <splat32> {value = 7 : i32} : i32
    ssavc4.vdw.store_vpm %addr, %y, %x, %active {
      elem_bytes = 4 : i32,
      row_len = 4 : i32,
      nrows = 1 : i32,
      memory_pitch_bytes = 16 : i32,
      orientation = "horizontal",
      serialize = "mutex"
    } : i32, i32, i32, i32
    ssavc4.thread_end
  }
}
