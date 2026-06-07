// RUN: not vc4-opt %s --split-input-file 2>&1 | FileCheck %s

ssavc4.module @bad_qpu_horizontal_dynamic_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    // CHECK: horizontal VPM QPU access does not encode a word x coordinate
    %read = ssavc4.vpm.read %row dynamic_x %x {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32 dynamic_x i32 -> vector<16xi32>
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_qpu_subword_dynamic_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    // CHECK: horizontal VPM QPU access does not encode a word x coordinate
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
    // CHECK: requires subword_selector attr or dynamic subword selector operand
    ssavc4.vdr.load %addr, %row dynamic_x %x {width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<packed>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32} : i32, i32 dynamic_x i32
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_qpu_subword_missing_selector {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    // CHECK: requires subword_selector attr or dynamic subword selector operand
    %read = ssavc4.vpm.read %row {width = #ssavc4.vpm_elem_width<w16>, subword = #ssavc4.vpm_subword<packed>, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32 -> vector<16xi32>
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_qpu_subword_both_selectors {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %sel = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %value = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    // CHECK: specify either static subword_selector or dynamic subword selector operand, not both
    ssavc4.vpm.write %row dynamic_subword_selector %sel, %value {width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<packed>, subword_selector = 0 : i32, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32 dynamic_subword_selector i32, vector<16xi32>
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_qpu_w32_dynamic_selector {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %sel = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    // CHECK: 32-bit VPM QPU access must not specify a dynamic subword selector
    %read = ssavc4.vpm.read %row dynamic_subword_selector %sel {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>} : i32 dynamic_subword_selector i32 -> vector<16xi32>
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
    ssavc4.vpm.write %row, %value {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<vertical>} : i32, vector<16xi32>
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
    %read = ssavc4.vpm.read %row {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, stride = 1 : i32, lanes = 16 : i32, orientation = #ssavc4.vpm_orientation<vertical>} : i32 -> vector<16xi32>
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

ssavc4.module @bad_vdr_load_both_selector {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %sel = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    // CHECK: specify either static subword_selector or dynamic subword selector operand, not both
    ssavc4.vdr.load %addr, %row dynamic_x %x dynamic_subword_selector %sel {width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<packed>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 16 : i32, subword_selector = 0 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32} : i32, i32 dynamic_x i32 dynamic_subword_selector i32
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vdr_load_w32_dynamic_selector {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %sel = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    // CHECK: 32-bit VPM DMA must not specify a dynamic subword selector
    ssavc4.vdr.load %addr, %row dynamic_x %x dynamic_subword_selector %sel {width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 64 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32} : i32, i32 dynamic_x i32 dynamic_subword_selector i32
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vdr_load_selector_out_of_range {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    // CHECK: subword_selector must be in range [0, 3]
    ssavc4.vdr.load %addr, %row dynamic_x %x {width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<packed>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 16 : i32, subword_selector = 4 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32} : i32, i32 dynamic_x i32
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vdr_rect_both_selector {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %sel = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %pitch = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    // CHECK: specify either static subword_selector or dynamic subword selector operand, not both
    ssavc4.vdr.load_rect.dynamic %addr, %row dynamic_dst_x %x dynamic_subword_selector %sel, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 2 : i32, orientation = #ssavc4.vpm_orientation<vertical>, width = #ssavc4.vpm_elem_width<w16>, subword = #ssavc4.vpm_subword<packed>, subword_selector = 0 : i32, vpm_pitch = 1 : i32, zero_fill = true} : i32, i32 dynamic_dst_x i32 dynamic_subword_selector i32, i32, i32, i32
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vdr_rect_neither_selector {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    %pitch = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    // CHECK: requires subword_selector attr or dynamic subword selector operand
    ssavc4.vdr.load_rect.dynamic %addr, %row dynamic_dst_x %x, %rows, %cols, %pitch {max_rows = 1 : i32, max_cols = 16 : i32, elem_bytes = 2 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w16>, subword = #ssavc4.vpm_subword<packed>, vpm_pitch = 1 : i32, zero_fill = true} : i32, i32 dynamic_dst_x i32, i32, i32, i32
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vdr_load_vector_selector {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %sel = ssavc4.load_imm <splat32> {value = 1 : i32} : vector<16xi32>
    // CHECK: requires an i32 VPM DMA dynamic subword selector operand
    ssavc4.vdr.load %addr, %row dynamic_x %x dynamic_subword_selector %sel {width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<packed>, row_len = 16 : i32, nrows = 1 : i32, memory_pitch_bytes = 16 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, vpm_pitch = 1 : i32} : i32, i32 dynamic_x i32 dynamic_subword_selector vector<16xi32>
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
