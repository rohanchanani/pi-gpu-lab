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
      warps_per_block_max = 1 : i32,
      uses_shared_vpm = false,
      uses_barrier = false,
      semaphores_per_block = 0 : i32,
      require_full_block_residency = false
    }
  } {
    %qpu_id = ssavc4.uniform.read 0 : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %sum = ssavc4.alu.add %qpu_id, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %vec = ssavc4.splat %sum : i32 -> vector<16xi32>
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    ssavc4.vdw.store %addr, %vec {elem_bytes = 4 : i32, active_lanes = 16 : i32, vpm_row = 0 : i32, serialize = "none"} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
