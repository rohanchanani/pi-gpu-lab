ssavc4.module @block_args_merge_smoke_ssavc4 {
  ssavc4.func @block_args_merge_smoke_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "block_args_merge_smoke_ssavc4",
      code_symbol = "block_args_merge_smoke_ssavc4_shader",
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
    %out = ssavc4.uniform.read 0 : i32
    %n = ssavc4.uniform.read 1 : i32
    %case_id = ssavc4.uniform.read 2 : i32
    %qpu_id = ssavc4.uniform.read 3 : i32

    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %shift_two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %shift_four = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %shift_six = ssavc4.load_imm <splat32> {value = 6 : i32} : i32
    %shift_eight = ssavc4.load_imm <splat32> {value = 8 : i32} : i32
    %active_full = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %full_bytes = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    %true_base = ssavc4.load_imm <splat32> {value = 4096 : i32} : vector<16xi32>
    %false_base = ssavc4.load_imm <splat32> {value = 8192 : i32} : vector<16xi32>
    %lane = ssavc4.element_number : vector<16xi32>

    %byte_offset = ssavc4.alu.add %qpu_id, %shift_six {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %logical_bytes = ssavc4.alu.add %n, %shift_two {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %skip_flags = ssavc4.make_flags %byte_offset, %logical_bytes {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %skip_flags, ^done, ^body {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^body:
    %addr = ssavc4.alu.add %out, %byte_offset {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %remaining = ssavc4.alu.add %logical_bytes, %byte_offset {opcode = #vc4.add_opcode<sub>} : (i32, i32) -> i32
    %base_elem = ssavc4.alu.add %qpu_id, %shift_four {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %base_vec = ssavc4.splat %base_elem : i32 -> vector<16xi32>
    %index = ssavc4.alu.add %base_vec, %lane {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %case_shift8 = ssavc4.alu.add %case_id, %shift_eight {opcode = #vc4.add_opcode<shl>} : (i32, i32) -> i32
    %case_vec = ssavc4.splat %case_shift8 : i32 -> vector<16xi32>
    %true_tag = ssavc4.alu.add %true_base, %case_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %true_value = ssavc4.alu.add %true_tag, %index {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %double_index = ssavc4.alu.add %index, %index {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %triple_index = ssavc4.alu.add %double_index, %index {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false_tag = ssavc4.alu.add %false_base, %case_vec {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %false_value = ssavc4.alu.add %false_tag, %triple_index {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %parity = ssavc4.alu.add %qpu_id, %one {opcode = #vc4.add_opcode<and>} : (i32, i32) -> i32
    %parity_flags = ssavc4.make_flags %parity, %zero {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %parity_flags, ^even(%true_value : vector<16xi32>), ^odd(%false_value : vector<16xi32>) {cond = #vc4.branch_cond<any_z_set>} : !ssavc4.flags

  ^odd(%odd_value: vector<16xi32>):
    ssavc4.br ^merge(%odd_value : vector<16xi32>)

  ^even(%even_value: vector<16xi32>):
    ssavc4.br ^merge(%even_value : vector<16xi32>)

  ^merge(%selected: vector<16xi32>):
    %full_flags = ssavc4.make_flags %remaining, %full_bytes {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %full_flags, ^full(%selected : vector<16xi32>), ^tail(%selected : vector<16xi32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^tail(%tail_value: vector<16xi32>):
    %active = ssavc4.alu.add %remaining, %shift_two {opcode = #vc4.add_opcode<shr>} : (i32, i32) -> i32
    ssavc4.vdw.store %addr, %tail_value, %active, %qpu_id {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32
    ssavc4.br ^done

  ^full(%full_value: vector<16xi32>):
    ssavc4.vdw.store %addr, %full_value, %active_full, %qpu_id {elem_bytes = 4 : i32, vpm_row = 0 : i32, serialize = "mutex"} : i32, vector<16xi32>, i32, i32
    ssavc4.br ^done

  ^done:
    ssavc4.thread_end
  }
}
