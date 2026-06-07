// RUN: not vc4-opt %s --split-input-file 2>&1 | FileCheck %s

ssavc4.module @bad_qpu_horizontal_dynamic_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    // CHECK: horizontal 32-bit VPM QPU access encodes Y only; dynamic x is not meaningful
    %read = ssavc4.vpm.read %row dynamic_x %x {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32 dynamic_x i32 -> vector<16xi32>
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_qpu_subword_dynamic_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    // CHECK: dynamic subword VPM QPU x selectors are unproven/deferred in P12
    %read = ssavc4.vpm.read %row dynamic_x %x {width = #ssavc4.vpm_elem_width<w16>, subword = #ssavc4.vpm_subword<packed>, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32 dynamic_x i32 -> vector<16xi32>
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_dma_subword_dynamic_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    // CHECK: dynamic subword VPM DMA x selectors are unproven/deferred in P12
    ssavc4.vdr.load %addr, %row dynamic_x %x {width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<packed>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32} : i32, i32 dynamic_x i32
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vpm_write_both_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    // CHECK: specify either static x or dynamic x operand, not both
    ssavc4.vpm.write %row dynamic_x %x, %value {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<vertical>} : i32 dynamic_x i32, vector<16xi32>
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vpm_write_neither_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    // CHECK: requires static x attr or dynamic x operand
    ssavc4.vpm.write %row, %value {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vpm_read_both_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    // CHECK: specify either static x or dynamic x operand, not both
    %read = ssavc4.vpm.read %row dynamic_x %x {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, x = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<vertical>} : i32 dynamic_x i32 -> vector<16xi32>
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vpm_read_neither_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    // CHECK: requires static x attr or dynamic x operand
    %read = ssavc4.vpm.read %row {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32 -> vector<16xi32>
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vdr_load_both_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    // CHECK: specify either static x or dynamic x operand, not both
    ssavc4.vdr.load %addr, %row dynamic_x %x {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 64 : i32, vpm_x = 0 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32} : i32, i32 dynamic_x i32
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vdr_load_neither_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    // CHECK: requires static x attr or dynamic x operand
    ssavc4.vdr.load %addr, %row {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 64 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32} : i32, i32
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vdr_rect_both_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %pitch = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    // CHECK: specify either static x or dynamic x operand, not both
    ssavc4.vdr.load_rect.dynamic %addr, %row dynamic_dst_x %x, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, dst_x = 0 : i32, vpm_pitch = 1 : i32, zero_fill = true} : i32, i32 dynamic_dst_x i32, i32, i32, i32
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vdr_rect_neither_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %pitch = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    // CHECK: requires static x attr or dynamic x operand
    ssavc4.vdr.load_rect.dynamic %addr, %row, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_pitch = 1 : i32, zero_fill = true} : i32, i32, i32, i32, i32
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vdw_rect_both_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %stride = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    // CHECK: specify either static x or dynamic x operand, not both
    ssavc4.vdw.store_rect.dynamic %addr, %row dynamic_src_x %x, %rows, %cols, %stride {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, src_x = 0 : i32, vpm_pitch = 1 : i32, preserve_inactive = true} : i32, i32 dynamic_src_x i32, i32, i32, i32
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vdw_rect_neither_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %stride = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    // CHECK: requires static x attr or dynamic x operand
    ssavc4.vdw.store_rect.dynamic %addr, %row, %rows, %cols, %stride {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 4 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, vpm_pitch = 1 : i32, preserve_inactive = true} : i32, i32, i32, i32, i32
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vdw_store_vpm_vector_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : vector<16xi32>
    // CHECK: requires an i32 VPM x-coordinate operand
    ssavc4.vdw.store_vpm %addr, %row, %x {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 64 : i32, active_lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32, i32, vector<16xi32>
    ssavc4.thread_end
  }
}
