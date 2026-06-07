// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: ssavc4.module @dynamic_vpm_coordinates
// CHECK: ssavc4.vpm.write %{{.*}} dynamic_x %{{.*}}, %{{.*}} {
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<vertical>
// CHECK-NOT: x =
// CHECK: ssavc4.vpm.read %{{.*}} dynamic_x %{{.*}} {
// CHECK: ssavc4.vpm.write %{{.*}} dynamic_subword_selector %{{.*}}, %{{.*}} {
// CHECK-SAME: width = #ssavc4.vpm_elem_width<w16>
// CHECK: ssavc4.vpm.read %{{.*}} dynamic_x %{{.*}} dynamic_subword_selector %{{.*}} {
// CHECK-SAME: width = #ssavc4.vpm_elem_width<w8>
// CHECK: ssavc4.vdr.load %{{.*}}, %{{.*}} dynamic_x %{{.*}} {
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<horizontal>
// CHECK: ssavc4.vdr.load %{{.*}}, %{{.*}} dynamic_x %{{.*}} {
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<vertical>
// CHECK: ssavc4.vdr.load %{{.*}}, %{{.*}} dynamic_x %{{.*}} dynamic_subword_selector %{{.*}} {
// CHECK-SAME: width = #ssavc4.vpm_elem_width<w8>
// CHECK: ssavc4.vdr.load_rect.dynamic %{{.*}}, %{{.*}} dynamic_dst_x %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<horizontal>
// CHECK: ssavc4.vdr.load_rect.dynamic %{{.*}}, %{{.*}} dynamic_dst_x %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<vertical>
// CHECK: ssavc4.vdr.load_rect.dynamic %{{.*}}, %{{.*}} dynamic_dst_x %{{.*}} dynamic_subword_selector %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
// CHECK-SAME: width = #ssavc4.vpm_elem_width<w16>
// CHECK: ssavc4.vdw.store_vpm %{{.*}}, %{{.*}}, %{{.*}} {
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<horizontal>
// CHECK: ssavc4.vdw.store_vpm %{{.*}}, %{{.*}}, %{{.*}} {
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<vertical>
// CHECK: ssavc4.vdw.store_rect.dynamic %{{.*}}, %{{.*}} dynamic_src_x %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<horizontal>
// CHECK: ssavc4.vdw.store_rect.dynamic %{{.*}}, %{{.*}} dynamic_src_x %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<vertical>
// CHECK-LABEL: ssavc4.module @static_vpm_coordinates
// CHECK: ssavc4.vpm.write %{{.*}}, %{{.*}} {
// CHECK-SAME: x = 0 : i32
// CHECK: ssavc4.vpm.read %{{.*}} {
// CHECK-SAME: x = 0 : i32
// CHECK: ssavc4.vdr.load %{{.*}}, %{{.*}} {
// CHECK-SAME: vpm_x = 3 : i32
// CHECK: ssavc4.vdr.load_rect.dynamic %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
// CHECK-SAME: dst_x = 3 : i32
// CHECK: ssavc4.vdw.store_rect.dynamic %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {
// CHECK-SAME: src_x = 3 : i32
ssavc4.module @dynamic_vpm_coordinates {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 4 : i32,
      user_vpm_rows_per_block = 16 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 16 : i32,
      uses_tmu = false,
      uses_vpm = true,
      uses_vpm_qpu_read = true,
      uses_vpm_qpu_write = true,
      uses_vdr = true,
      uses_vdw = true,
      uses_barrier = true,
      semaphore_count_per_block = 4 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = true
    }
  } {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %sel = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %pitch = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    ssavc4.vpm.write %row dynamic_x %x, %value {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<vertical>} : i32 dynamic_x i32, vector<16xi32>
    %read = ssavc4.vpm.read %row dynamic_x %x {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<vertical>} : i32 dynamic_x i32 -> vector<16xi32>
    ssavc4.vpm.write %row dynamic_subword_selector %sel, %value {width = #ssavc4.vpm_elem_width<w16>, subword = #ssavc4.vpm_subword<packed>, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32 dynamic_subword_selector i32, vector<16xi32>
    %read_sub = ssavc4.vpm.read %row dynamic_x %x dynamic_subword_selector %sel {width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<laned>, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<vertical>} : i32 dynamic_x i32 dynamic_subword_selector i32 -> vector<16xi32>
    ssavc4.vdr.load %addr, %row dynamic_x %x {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 64 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32} : i32, i32 dynamic_x i32
    ssavc4.vdr.load %addr, %row dynamic_x %x {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 64 : i32, orientation = #ssavc4.vpm_orientation<vertical>, vpm_pitch = 1 : i32} : i32, i32 dynamic_x i32
    ssavc4.vdr.load %addr, %row dynamic_x %x dynamic_subword_selector %sel {width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<packed>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32} : i32, i32 dynamic_x i32 dynamic_subword_selector i32
    ssavc4.vdr.load_rect.dynamic %addr, %row dynamic_dst_x %x, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_pitch = 1 : i32, zero_fill = true} : i32, i32 dynamic_dst_x i32, i32, i32, i32
    ssavc4.vdr.load_rect.dynamic %addr, %row dynamic_dst_x %x, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #ssavc4.vpm_orientation<vertical>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_pitch = 1 : i32, zero_fill = true} : i32, i32 dynamic_dst_x i32, i32, i32, i32
    ssavc4.vdr.load_rect.dynamic %addr, %row dynamic_dst_x %x dynamic_subword_selector %sel, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 2 : i32, orientation = #ssavc4.vpm_orientation<vertical>, width = #ssavc4.vpm_elem_width<w16>, subword = #ssavc4.vpm_subword<packed>, vpm_pitch = 1 : i32, zero_fill = true} : i32, i32 dynamic_dst_x i32 dynamic_subword_selector i32, i32, i32, i32
    ssavc4.vdw.store_vpm %addr, %row, %x {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 64 : i32, active_lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32, i32, i32
    ssavc4.vdw.store_vpm %addr, %row, %x {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 64 : i32, active_lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<vertical>} : i32, i32, i32
    ssavc4.vdw.store_rect.dynamic %addr, %row dynamic_src_x %x, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_pitch = 1 : i32, preserve_inactive = true} : i32, i32 dynamic_src_x i32, i32, i32, i32
    ssavc4.vdw.store_rect.dynamic %addr, %row dynamic_src_x %x, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #ssavc4.vpm_orientation<vertical>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_pitch = 1 : i32, preserve_inactive = true} : i32, i32 dynamic_src_x i32, i32, i32, i32
    ssavc4.thread_end
  }
}

ssavc4.module @static_vpm_coordinates {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 4 : i32,
      user_vpm_rows_per_block = 16 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 16 : i32,
      uses_tmu = false,
      uses_vpm = true,
      uses_vpm_qpu_read = true,
      uses_vpm_qpu_write = true,
      uses_vdr = true,
      uses_vdw = true,
      uses_barrier = true,
      semaphore_count_per_block = 4 : i32,
      requires_vpm_base_row_builtin = true,
      requires_semaphore_base_builtin = true
    }
  } {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %pitch = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    ssavc4.vpm.write %row, %value {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32, vector<16xi32>
    %read = ssavc4.vpm.read %row {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32 -> vector<16xi32>
    ssavc4.vdr.load %addr, %row {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 64 : i32, vpm_x = 3 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32} : i32, i32
    ssavc4.vdr.load_rect.dynamic %addr, %row, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, dst_x = 3 : i32, vpm_pitch = 1 : i32, zero_fill = true} : i32, i32, i32, i32, i32
    ssavc4.vdw.store_vpm %addr, %row, %x {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 64 : i32, active_lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32, i32, i32
    ssavc4.vdw.store_rect.dynamic %addr, %row, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, src_x = 3 : i32, vpm_pitch = 1 : i32, preserve_inactive = true} : i32, i32, i32, i32, i32
    ssavc4.thread_end
  }
}
