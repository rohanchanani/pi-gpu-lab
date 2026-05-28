// RUN: not vc4-opt %s --convert-ssavc4-to-vc4 -o /dev/null 2>&1 | FileCheck %s

ssavc4.module @bad_vdr_load_lowering {
  ssavc4.func @bad_vdr_load_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "bad_vdr_load_lowering",
      code_symbol = "bad_vdr_load_lowering_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 0 : i32,
      args = [],
      builtins = []
    },
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      uses_barrier = false,
      uses_shared_vpm = false,
      require_full_block_residency = false,
      semaphores_per_block = 0 : i32,
      warps_per_block_max = 1 : i32
    }
  } {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    // CHECK: requires vc4.resource schedule_mode = cooperative_block
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
