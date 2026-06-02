// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: ssavc4.module @dynamic_vdr_vdw_rect
// CHECK: ssavc4.vdr.load_rect.dynamic
// CHECK-SAME: max_cols = 8 : i32
// CHECK-SAME: max_rows = 4 : i32
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<horizontal>
// CHECK-SAME: width = #ssavc4.vpm_elem_width<w32>
// CHECK-SAME: zero_fill = true
// CHECK: ssavc4.vdw.store_rect.dynamic
// CHECK-SAME: max_cols = 8 : i32
// CHECK-SAME: max_rows = 4 : i32
// CHECK-SAME: orientation = #ssavc4.vpm_orientation<vertical>
// CHECK-SAME: preserve_inactive = true
ssavc4.module @dynamic_vdr_vdw_rect {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %rows = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %cols = ssavc4.load_imm <splat32> {value = 5 : i32} : i32
    %pitch = ssavc4.load_imm <splat32> {value = 64 : i32} : i32
    ssavc4.vdr.load_rect.dynamic %addr, %row, %rows, %cols, %pitch {
      max_rows = 4 : i32, max_cols = 8 : i32, elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<horizontal>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      dst_x = 0 : i32, vpm_pitch = 1 : i32,
      zero_fill = true, serialize = "mutex"
    } : i32, i32, i32, i32, i32
    ssavc4.vdw.store_rect.dynamic %addr, %row, %rows, %cols, %pitch {
      max_rows = 4 : i32, max_cols = 8 : i32, elem_bytes = 4 : i32,
      orientation = #ssavc4.vpm_orientation<vertical>,
      width = #ssavc4.vpm_elem_width<w32>,
      subword = #ssavc4.vpm_subword<none>,
      src_x = 3 : i32, vpm_pitch = 1 : i32,
      preserve_inactive = true, serialize = "mutex"
    } : i32, i32, i32, i32, i32
    ssavc4.thread_end
  }
}
