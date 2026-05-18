ssavc4.module @vector_store_smoke_ssavc4 {
  ssavc4.func @vector_store_smoke_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      uses_barrier = false,
      uses_shared_vpm = false,
      require_full_block_residency = false,
      semaphores_per_block = 0 : i32,
      warps_per_block_max = 1 : i32
    },
    "vc4.launch_abi" = {
      public_name = "vector_store_smoke_ssavc4",
      code_symbol = "vector_store_smoke_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 5 : i32,
      args = [
        {name = "out", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 0 : i32},
        {name = "n", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 1 : i32},
        {name = "case_id", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32}
      ],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 3 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 4 : i32}
      ]
    }
  } {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    ssavc4.vdw.store %addr, %value {elem_bytes = 4 : i32, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
