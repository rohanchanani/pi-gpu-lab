// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @vdr_load_lowering
// CHECK: vc4.func @vdr_load_kernel
// CHECK: uses_shared_vpm = true
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_addr
// CHECK-SAME: side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_wait
// CHECK-SAME: side = #vc4.vpmvcd_side<read>
// CHECK: sig = #vc4.qpu_signal<thrend>
// CHECK-NOT: ssavc4.
ssavc4.module @vdr_load_lowering {
  ssavc4.func @vdr_load_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "vdr_load_lowering",
      code_symbol = "vdr_load_lowering_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 1 : i32,
      args = [
        {name = "in", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32}
      ],
      builtins = []
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      uses_barrier = true,
      uses_shared_vpm = true,
      shared_vpm_bytes = 1024 : i32,
      require_full_block_residency = true,
      semaphores_per_block = 4 : i32,
      warps_per_block_max = 4 : i32
    }
  } {
    %addr = ssavc4.uniform.read 0 : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    ssavc4.vdr.load %addr, %row {
      elem_bytes = 4 : i32,
      row_len = 16 : i32,
      nrows = 16 : i32,
      memory_pitch_bytes = 64 : i32,
      vpm_base_col = 0 : i32,
      orientation = "horizontal",
      vpitch = 1 : i32,
      serialize = "mutex"
    } : i32, i32
    ssavc4.thread_end
  }
}
