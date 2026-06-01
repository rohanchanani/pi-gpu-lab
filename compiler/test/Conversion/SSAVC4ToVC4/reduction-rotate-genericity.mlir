// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @neutral_reduce_module
// CHECK: vc4.func @neutral_reduce_kernel
// CHECK: public_name = "neutral_reduce_entry"
// CHECK: raddr_a = 38 : i32
// CHECK: waddr_add = 34 : i32
// CHECK: sig = #vc4.qpu_signal<small_imm>
// CHECK-SAME: small_imm = 52 : i32
// CHECK: pack = #vc4.regfile_a_pack_mode<to_16a>
// CHECK: op_add = #vc4.add_opcode<add>
// CHECK: sig = #vc4.qpu_signal<thrend>
// CHECK-NOT: ssavc4.
ssavc4.module @neutral_reduce_module {
  ssavc4.func @neutral_reduce_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "neutral_reduce_entry",
      code_symbol = "neutral_reduce_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 0 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 0 : i32,
      uses_tmu = false,
      uses_vpm = false,
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = false,
      uses_vdr = false,
      uses_vdw = false,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = false,
      requires_semaphore_base_builtin = false
    }
  } {
    %lane = ssavc4.element_number : vector<16xi32>
    %rot = ssavc4.rotate %lane {amount = 4 : i32} : vector<16xi32> -> vector<16xi32>
    %packed = ssavc4.pack %rot {mode = #vc4.regfile_a_pack_mode<to_16a>} : vector<16xi32> -> vector<16xi32>
    %wide = ssavc4.unpack %packed : vector<16xi32> -> vector<16xi32>
    %sum = ssavc4.alu.add %lane, %wide {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.thread_end
  }
}
