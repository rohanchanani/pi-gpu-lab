// RUN: not vc4-opt %s 2>&1 | FileCheck %s

ssavc4.module @bad_dynamic_rect_x {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %pitch = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    // CHECK: requires VPM x coordinate in range [0, 15]
    ssavc4.vdr.load_rect.dynamic %addr, %row, %rows, %cols, %pitch {max_rows = 4 : i32, max_cols = 4 : i32, elem_bytes = 4 : i32, orientation = #ssavc4.vpm_orientation<vertical>, width = #ssavc4.vpm_elem_width<w32>, subword = #ssavc4.vpm_subword<none>, dst_x = 16 : i32, vpm_pitch = 1 : i32, zero_fill = true} : i32, i32, i32, i32, i32
    ssavc4.thread_end
  }
}
