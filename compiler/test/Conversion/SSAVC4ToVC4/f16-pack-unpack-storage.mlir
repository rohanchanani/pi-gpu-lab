// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @f16_pack_unpack_storage_lowering
// CHECK: vc4.qpu.bundle
// CHECK-SAME: op_add = #vc4.add_opcode<fadd>
// CHECK-SAME: sig = #vc4.qpu_signal<small_imm>
// CHECK-SAME: small_imm = 0 : i32
// CHECK-SAME: unpack = #vc4.regfile_a_unpack_mode<f16a_or_i16a>
// CHECK: vc4.qpu.ldi <splat32> {cond_add = #vc4.cond<always>{{.*}}value = 0 : i32, waddr_add = [[PACK_CARRIER:[0-9]+]] : i32
// CHECK: vc4.qpu.bundle
// CHECK-SAME: op_add = #vc4.add_opcode<fadd>
// CHECK-SAME: pack = #vc4.regfile_a_pack_mode<to_16a>
// CHECK-SAME: sig = #vc4.qpu_signal<small_imm>
// CHECK-SAME: small_imm = 0 : i32
// CHECK-SAME: waddr_add = [[PACK_CARRIER]] : i32
// CHECK-NOT: ssavc4.
ssavc4.module @f16_pack_unpack_storage_lowering {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {public_name = "f16_pack_unpack_storage_lowering", code_symbol = "f16_pack_unpack_storage_lowering_shader", tail_policy = "exact_multiple", uniform_words_per_qpu = 2 : i32, args = [], builtins = [{name = "logical_request", kind = #vc4.builtin_kind<logical_request>, materialization = "uniform_suffix", uniform_index = 0 : i32}, {name = "total_requests", kind = #vc4.builtin_kind<total_requests>, materialization = "uniform_suffix", uniform_index = 1 : i32}]},
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
    %carrier = ssavc4.load_imm <splat32> {value = 15360 : i32} : vector<16xi32>
    %f = ssavc4.unpack %carrier {f16_storage_conversion, mode = #vc4.regfile_a_unpack_mode<f16a_or_i16a>} : vector<16xi32> -> vector<16xf32>
    %packed = ssavc4.pack %f {f16_storage_conversion, mode = #vc4.regfile_a_pack_mode<to_16a>} : vector<16xf32> -> vector<16xi32>
    ssavc4.thread_end
  }
}
