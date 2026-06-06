// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @pack_branch_layout
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_c_clear>, immediate = 64 : i32
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<always>, immediate = 160 : i32
// CHECK: value = 4660 : i32, waddr_add = 0 : i32
// CHECK: value = 0 : i32, waddr_add = 1 : i32
// CHECK-NEXT: vc4.qpu.bundle
// CHECK-SAME: pack = #vc4.regfile_a_pack_mode<to_16a>
// CHECK-SAME: raddr_a = 0 : i32
// CHECK-SAME: waddr_add = 1 : i32
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<always>, immediate = 72 : i32
// CHECK: value = 22136 : i32, waddr_add = 0 : i32
ssavc4.module @pack_branch_layout {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {public_name = "pack_branch_layout", code_symbol = "pack_branch_layout_shader", tail_policy = "exact_multiple", uniform_words_per_qpu = 2 : i32, args = [], builtins = [{name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32}, {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 1 : i32}]},
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
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %flags = ssavc4.make_flags %one, %zero {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %flags, ^pack_path, ^skip {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags
  ^pack_path:
    %src = ssavc4.load_imm <splat32> {value = 4660 : i32} : vector<16xi32>
    %p = ssavc4.pack %src {mode = #vc4.regfile_a_pack_mode<to_16a>} : vector<16xi32> -> vector<16xi32>
    %u = ssavc4.unpack %p : vector<16xi32> -> vector<16xi32>
    %sink = ssavc4.alu.add %u, %src {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.br ^done
  ^skip:
    %fallback = ssavc4.load_imm <splat32> {value = 22136 : i32} : vector<16xi32>
    ssavc4.br ^done
  ^done:
    ssavc4.thread_end
  }
}
