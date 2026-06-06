// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s
// CHECK-LABEL: vc4.func @vpm_subword_setup
// w8 horizontal packed x=3: ID | STRIDE(1) | HORIZ | SIZE(8) | ADDR(3)
// CHECK: value = 1054723 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// w16 horizontal laned x=1: ID | STRIDE(1) | HORIZ | LANED | SIZE(16) | ADDR(1)
// CHECK: value = 1056001 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// w16 vertical packed x=1: ID | STRIDE(1) | SIZE(16) | ADDR(1)
// CHECK: value = 1052929 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// w8 vertical laned x=2: ID | STRIDE(1) | LANED | SIZE(8) | ADDR(2)
// CHECK: value = 1053698 : i32
// CHECK: vc4.qpu.vpmvcd_setup
ssavc4.module @vpm_subword_ssavc4 {
  ssavc4.func @vpm_subword_setup() attributes {
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
      uses_vpm_qpu_write = true,
      uses_vdr = false,
      uses_vdw = false,
      uses_barrier = false,
      semaphore_count_per_block = 0 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = false
    }
  } {
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    ssavc4.vpm.write %row0, %value {width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<packed>, x = 3 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32, vector<16xi32>
    ssavc4.vpm.write %row0, %value {width = #ssavc4.vpm_elem_width<w16>, subword = #ssavc4.vpm_subword<laned>, x = 1 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32, vector<16xi32>
    %read0 = ssavc4.vpm.read %row0 {width = #ssavc4.vpm_elem_width<w16>, subword = #ssavc4.vpm_subword<packed>, x = 1 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<vertical>} : i32 -> vector<16xi32>
    %read1 = ssavc4.vpm.read %row0 {width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<laned>, x = 2 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<vertical>} : i32 -> vector<16xi32>
    ssavc4.thread_end
  }
}
