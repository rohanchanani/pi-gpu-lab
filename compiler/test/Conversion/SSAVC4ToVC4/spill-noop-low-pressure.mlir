// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.func @spill_noop_low_pressure_kernel
// CHECK-NOT: spill_frame_bytes
// CHECK-NOT: spill_frame_base
// CHECK: sig = #vc4.qpu_signal<thrend>
ssavc4.module @spill_noop_low_pressure {
  ssavc4.func @spill_noop_low_pressure_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "spill_noop_low_pressure",
      code_symbol = "spill_noop_low_pressure_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 1 : i32,
      args = [],
      builtins = [
        {name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 0 : i32,
      compiler_vpm_staging_rows_per_warp = 1 : i32,
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
    %qpu_id = ssavc4.uniform.read 0 : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %sum = ssavc4.alu.add %qpu_id, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %vec = ssavc4.splat %sum : i32 -> vector<16xi32>
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    ssavc4.vdw.store %addr, %vec {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "none"} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
