// RUN: vc4-opt %s --convert-ssavc4-to-vc4 --vc4-verify-scheduled-hardware-rules | FileCheck %s

// CHECK-LABEL: vc4.module @dynamic_rotate_lowering
// CHECK: op_add = #vc4.add_opcode<xor>
// CHECK-SAME: sig = #vc4.qpu_signal<small_imm>
// CHECK-SAME: small_imm = 15 : i32
// CHECK-SAME: waddr_add = 34 : i32
// CHECK-NEXT: vc4.qpu.bundle
// CHECK-SAME: cond_add = #vc4.cond<never>
// CHECK-SAME: op_add = #vc4.add_opcode<nop>
// CHECK-NEXT: vc4.qpu.bundle
// CHECK-SAME: op_add = #vc4.add_opcode<add>
// CHECK-SAME: sig = #vc4.qpu_signal<small_imm>
// CHECK-SAME: small_imm = 1 : i32
// CHECK-SAME: waddr_add = 37 : i32
// CHECK-NEXT: vc4.qpu.bundle
// CHECK-SAME: cond_add = #vc4.cond<never>
// CHECK-SAME: op_add = #vc4.add_opcode<nop>
// CHECK-NEXT: vc4.qpu.bundle
// CHECK-SAME: waddr_add = 34 : i32
// CHECK-NEXT: vc4.qpu.bundle
// CHECK-SAME: cond_add = #vc4.cond<never>
// CHECK-SAME: op_add = #vc4.add_opcode<nop>
// CHECK-NEXT: vc4.qpu.bundle
// CHECK-SAME: sig = #vc4.qpu_signal<small_imm>
// CHECK-SAME: small_imm = 48 : i32
// CHECK: sig = #vc4.qpu_signal<thrend>
ssavc4.module @dynamic_rotate_lowering {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {public_name = "dynamic_rotate_lowering", code_symbol = "dynamic_rotate_lowering_shader", tail_policy = "exact_multiple", uniform_words_per_qpu = 2 : i32, args = [], builtins = [{name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32}, {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 1 : i32}]},
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
    %amount = ssavc4.load_imm <splat32> {value = 17 : i32} : i32
    %v = ssavc4.element_number : vector<16xi32>
    %r = ssavc4.rotate %v, %amount : vector<16xi32>, i32 -> vector<16xi32>
    %z = ssavc4.alu.add %r, %v {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.thread_end
  }
}

// CHECK-LABEL: vc4.module @static_rotate_zero_lowering
// CHECK-LABEL: vc4.func @kernel
// CHECK: waddr_add = 32 : i32
// CHECK-NOT: small_imm = 48 : i32
// CHECK: sig = #vc4.qpu_signal<thrend>
ssavc4.module @static_rotate_zero_lowering {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {public_name = "static_rotate_zero_lowering", code_symbol = "static_rotate_zero_lowering_shader", tail_policy = "exact_multiple", uniform_words_per_qpu = 2 : i32, args = [], builtins = [{name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32}, {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 1 : i32}]},
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
    %v = ssavc4.element_number : vector<16xi32>
    %r = ssavc4.rotate %v {amount = 0 : i32} : vector<16xi32> -> vector<16xi32>
    %z = ssavc4.alu.add %r, %v {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.thread_end
  }
}
