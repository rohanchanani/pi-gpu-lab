// RUN: not vc4-opt %s 2>&1 | FileCheck %s

ssavc4.module @bad_dynamic_rect_subword {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %stride = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    // CHECK: 32-bit VDW DMA requires subword = #ssavc4.vpm_subword<none>
    ssavc4.vdw.store_rect.dynamic %addr, %row, %rows, %cols, %stride {max_rows = 4 : i32, max_cols = 4 : i32, elem_bytes = 4 : i32, orientation = #ssavc4.vpm_orientation<horizontal>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<packed>, src_x = 0 : i32, vpm_pitch = 1 : i32, preserve_inactive = true} : i32, i32, i32, i32, i32
    ssavc4.thread_end
  }
}

// -----

ssavc4.module @bad_vertical_subword_dma {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %stride = ssavc4.load_imm <splat32> {value = 16 : i32} : i32
    // CHECK: vertical subword VDW DMA is unproven/deferred in P12
    ssavc4.vdw.store_rect.dynamic %addr, %row, %rows, %cols, %stride {max_rows = 1 : i32, max_cols = 4 : i32, elem_bytes = 1 : i32, orientation = #ssavc4.vpm_orientation<vertical>, width = #ssavc4.vpm_elem_width<w8>, subword = #ssavc4.vpm_subword<packed>, src_x = 0 : i32, vpm_pitch = 1 : i32, preserve_inactive = true} : i32, i32, i32, i32, i32
    ssavc4.thread_end
  }
}
