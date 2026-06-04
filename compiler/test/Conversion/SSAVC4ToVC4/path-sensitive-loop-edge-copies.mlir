// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.func @loop_ifelse_vector_args_kernel
// CHECK-SAME: spill_frame_bytes =
// CHECK: vc4.qpu.branch attributes
// CHECK: vc4.qpu.branch attributes {{.*}}cond = #vc4.branch_cond<always>{{.*}}immediate = -{{[0-9]+}} : i32

// CHECK-LABEL: vc4.func @static_row_pingpong_branch_form_kernel
// CHECK: vc4.qpu.vpmvcd_setup {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.vpmvcd_wait {{.*}}side = #vc4.vpmvcd_side<read>
// CHECK: vc4.qpu.branch attributes {{.*}}cond = #vc4.branch_cond<always>{{.*}}immediate = -{{[0-9]+}} : i32

ssavc4.module @path_sensitive_loop_edge_copies {
  ssavc4.func @loop_ifelse_vector_args_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>
  } {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %acc0 = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
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
    ssavc4.br ^loop(%zero, %acc0 : i32, vector<16xi32>)

  ^loop(%i: i32, %acc: vector<16xi32>):
    %done_flags = ssavc4.make_flags %i, %two {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %done_flags, ^exit(%acc : vector<16xi32>), ^choose(%i, %acc : i32, vector<16xi32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^choose(%i_choose: i32, %acc_choose: vector<16xi32>):
    %row_flags = ssavc4.make_flags %i_choose, %zero {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %row_flags, ^row0(%i_choose, %acc_choose : i32, vector<16xi32>), ^row1(%i_choose, %acc_choose : i32, vector<16xi32>) {cond = #vc4.branch_cond<any_z_set>} : !ssavc4.flags

  ^row0(%i_row0: i32, %acc_row0: vector<16xi32>):
    %sum0 = ssavc4.alu.add %acc_row0, %v01 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.br ^merge(%i_row0, %sum0 : i32, vector<16xi32>)

  ^row1(%i_row1: i32, %acc_row1: vector<16xi32>):
    %sum1 = ssavc4.alu.add %acc_row1, %v02 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.br ^merge(%i_row1, %sum1 : i32, vector<16xi32>)

  ^merge(%i_merge: i32, %merged: vector<16xi32>):
    %i_next = ssavc4.alu.add %i_merge, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^loop(%i_next, %merged : i32, vector<16xi32>)

  ^exit(%final: vector<16xi32>):
    %s03 = ssavc4.alu.add %final, %v03 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s04 = ssavc4.alu.add %s03, %v04 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s05 = ssavc4.alu.add %s04, %v05 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s06 = ssavc4.alu.add %s05, %v06 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s07 = ssavc4.alu.add %s06, %v07 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s08 = ssavc4.alu.add %s07, %v08 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s09 = ssavc4.alu.add %s08, %v09 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s10 = ssavc4.alu.add %s09, %v10 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s11 = ssavc4.alu.add %s10, %v11 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s12 = ssavc4.alu.add %s11, %v12 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s13 = ssavc4.alu.add %s12, %v13 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s14 = ssavc4.alu.add %s13, %v14 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s15 = ssavc4.alu.add %s14, %v15 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s16 = ssavc4.alu.add %s15, %v16 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s17 = ssavc4.alu.add %s16, %v17 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s18 = ssavc4.alu.add %s17, %v18 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s19 = ssavc4.alu.add %s18, %v19 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s20 = ssavc4.alu.add %s19, %v20 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s21 = ssavc4.alu.add %s20, %v21 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s22 = ssavc4.alu.add %s21, %v22 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %s23 = ssavc4.alu.add %s22, %v23 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %sink = ssavc4.alu.add %s23, %v24 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.thread_end
  }

  ssavc4.func @static_row_pingpong_branch_form_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "independent_vector",
      warps_per_block = 1 : i32,
      user_vpm_rows_per_block = 4 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 4 : i32,
      uses_tmu = false,
      uses_vpm = true,
      uses_vpm_qpu_read = true,
      uses_vpm_qpu_write = false,
      uses_vdr = false,
      uses_vdw = false,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row2 = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    %acc0 = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    ssavc4.br ^loop(%zero, %acc0 : i32, vector<16xi32>)

  ^loop(%i: i32, %acc: vector<16xi32>):
    %done_flags = ssavc4.make_flags %i, %two {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %done_flags, ^exit(%acc : vector<16xi32>), ^select(%i, %acc : i32, vector<16xi32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^select(%i_select: i32, %acc_select: vector<16xi32>):
    %row_flags = ssavc4.make_flags %i_select, %zero {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %row_flags, ^load0(%i_select, %acc_select : i32, vector<16xi32>), ^load2(%i_select, %acc_select : i32, vector<16xi32>) {cond = #vc4.branch_cond<any_z_set>} : !ssavc4.flags

  ^load0(%i_load0: i32, %acc0_path: vector<16xi32>):
    %r0 = ssavc4.vpm.read %row0 {lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, stride = 1 : i32, subword = #ssavc4.vpm_subword<none>, width = #ssavc4.vpm_elem_width<w32>, x = 0 : i32} : i32 -> vector<16xi32>
    %sum0 = ssavc4.alu.add %acc0_path, %r0 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.br ^merge(%i_load0, %sum0 : i32, vector<16xi32>)

  ^load2(%i_load2: i32, %acc2_path: vector<16xi32>):
    %r2 = ssavc4.vpm.read %row2 {lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, stride = 1 : i32, subword = #ssavc4.vpm_subword<none>, width = #ssavc4.vpm_elem_width<w32>, x = 0 : i32} : i32 -> vector<16xi32>
    %sum2 = ssavc4.alu.add %acc2_path, %r2 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.br ^merge(%i_load2, %sum2 : i32, vector<16xi32>)

  ^merge(%i_merge: i32, %merged: vector<16xi32>):
    %i_next = ssavc4.alu.add %i_merge, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^loop(%i_next, %merged : i32, vector<16xi32>)

  ^exit(%final: vector<16xi32>):
    %sink = ssavc4.alu.add %final, %acc0 {opcode = #vc4.add_opcode<add>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.thread_end
  }
}
