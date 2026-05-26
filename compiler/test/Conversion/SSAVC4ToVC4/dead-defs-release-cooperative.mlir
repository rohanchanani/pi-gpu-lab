// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck --implicit-check-not=spill_frame_bytes --implicit-check-not=spill_frame_base %s

// CHECK-LABEL: vc4.func @dead_defs_release_cooperative_kernel
// CHECK-SAME: public_name = "dead_defs_release_cooperative"
// CHECK: value = 32 : i32
// CHECK: sig = #vc4.qpu_signal<thrend>
ssavc4.module @dead_defs_release_cooperative {
  ssavc4.func @dead_defs_release_cooperative_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "dead_defs_release_cooperative",
      code_symbol = "dead_defs_release_cooperative_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 1 : i32,
      args = [],
      builtins = [
        {name = "logical_warp_id", kind = #vc4.builtin_kind<logical_warp_id>, materialization = "uniform_suffix", uniform_index = 0 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block_max = 2 : i32,
      uses_shared_vpm = false,
      uses_barrier = false,
      semaphores_per_block = 0 : i32,
      require_full_block_residency = true
    }
  } {
    %qpu_id = ssavc4.uniform.read 0 : i32
    %lane = ssavc4.element_number : vector<16xi32>
    %base = ssavc4.splat %qpu_id : i32 -> vector<16xi32>
    %v00 = ssavc4.alu.add %base, %lane {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %v01 = ssavc4.load_imm <splat32> {value = 1 : i32} : vector<16xi32>
    %v02 = ssavc4.load_imm <splat32> {value = 2 : i32} : vector<16xi32>
    %v03 = ssavc4.load_imm <splat32> {value = 3 : i32} : vector<16xi32>
    %v04 = ssavc4.load_imm <splat32> {value = 4 : i32} : vector<16xi32>
    %v05 = ssavc4.load_imm <splat32> {value = 5 : i32} : vector<16xi32>
    %v06 = ssavc4.load_imm <splat32> {value = 6 : i32} : vector<16xi32>
    %v07 = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    %v08 = ssavc4.load_imm <splat32> {value = 8 : i32} : vector<16xi32>
    %v09 = ssavc4.load_imm <splat32> {value = 9 : i32} : vector<16xi32>
    %v10 = ssavc4.load_imm <splat32> {value = 10 : i32} : vector<16xi32>
    %v11 = ssavc4.load_imm <splat32> {value = 11 : i32} : vector<16xi32>
    %v12 = ssavc4.load_imm <splat32> {value = 12 : i32} : vector<16xi32>
    %v13 = ssavc4.load_imm <splat32> {value = 13 : i32} : vector<16xi32>
    %v14 = ssavc4.load_imm <splat32> {value = 14 : i32} : vector<16xi32>
    %v15 = ssavc4.load_imm <splat32> {value = 15 : i32} : vector<16xi32>
    %v16 = ssavc4.load_imm <splat32> {value = 16 : i32} : vector<16xi32>
    %v17 = ssavc4.load_imm <splat32> {value = 17 : i32} : vector<16xi32>
    %v18 = ssavc4.load_imm <splat32> {value = 18 : i32} : vector<16xi32>
    %v19 = ssavc4.load_imm <splat32> {value = 19 : i32} : vector<16xi32>
    %v20 = ssavc4.load_imm <splat32> {value = 20 : i32} : vector<16xi32>
    %v21 = ssavc4.load_imm <splat32> {value = 21 : i32} : vector<16xi32>
    %v22 = ssavc4.load_imm <splat32> {value = 22 : i32} : vector<16xi32>
    %v23 = ssavc4.load_imm <splat32> {value = 23 : i32} : vector<16xi32>
    %v24 = ssavc4.load_imm <splat32> {value = 24 : i32} : vector<16xi32>
    %v25 = ssavc4.load_imm <splat32> {value = 25 : i32} : vector<16xi32>
    %v26 = ssavc4.load_imm <splat32> {value = 26 : i32} : vector<16xi32>
    %v27 = ssavc4.load_imm <splat32> {value = 27 : i32} : vector<16xi32>
    %v28 = ssavc4.load_imm <splat32> {value = 28 : i32} : vector<16xi32>
    %v29 = ssavc4.load_imm <splat32> {value = 29 : i32} : vector<16xi32>
    %v30 = ssavc4.load_imm <splat32> {value = 30 : i32} : vector<16xi32>
    %v31 = ssavc4.load_imm <splat32> {value = 31 : i32} : vector<16xi32>
    %v32 = ssavc4.load_imm <splat32> {value = 32 : i32} : vector<16xi32>
    %sum = ssavc4.alu.add %v00, %v32 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.thread_end
  }
}
