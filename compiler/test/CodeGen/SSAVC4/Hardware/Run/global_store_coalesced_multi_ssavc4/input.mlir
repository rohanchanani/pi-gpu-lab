ssavc4.module @global_store_coalesced_multi_ssavc4 {
  ssavc4.func @global_store_coalesced_multi_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "global_store_coalesced_multi_ssavc4",
      tail_policy = "tail_safe",
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
    },
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      warps_per_block_max = 1 : i32,
      uses_shared_vpm = false,
      uses_barrier = false,
      semaphores_per_block = 0 : i32,
      require_full_block_residency = false
    }
  } {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    ssavc4.vdw.store %addr, %value {
      elem_bytes = 4 : i32,
      active_lanes = 16 : i32,
      vpm_row = 0 : i32,
      serialize = "mutex",
      lowering_template = "global_store_coalesced_multi_tail_u32"
    } : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
