// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @rotate_lowering
// CHECK: waddr_add = 34 : i32
// CHECK: sig = #vc4.qpu_signal<small_imm>
// CHECK-SAME: small_imm = 56 : i32
// CHECK: pack = #vc4.regfile_a_pack_mode<to_16a>
// CHECK: sig = #vc4.qpu_signal<thrend>
ssavc4.module @rotate_lowering {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {public_name = "rotate_lowering", code_symbol = "rotate_lowering_shader", tail_policy = "exact_multiple", uniform_words_per_qpu = 2 : i32, args = [], builtins = [{name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32}, {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 1 : i32}]},
    "vc4.resource" = {schedule_mode = "independent_vector", warps_per_block_max = 1 : i32, uses_shared_vpm = false, uses_barrier = false, semaphores_per_block = 0 : i32, require_full_block_residency = false}
  } {
    %v = ssavc4.load_imm <splat32> {value = 1 : i32} : vector<16xi32>
    %r = ssavc4.rotate %v {amount = 8 : i32} : vector<16xi32> -> vector<16xi32>
    %p = ssavc4.pack %r {mode = #vc4.regfile_a_pack_mode<to_16a>} : vector<16xi32> -> vector<16xi32>
    %u = ssavc4.unpack %p : vector<16xi32> -> vector<16xi32>
    %z = ssavc4.alu.add %u, %v {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.thread_end
  }
}
