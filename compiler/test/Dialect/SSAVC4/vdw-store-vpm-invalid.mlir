// RUN: not vc4-opt %s -o /dev/null 2>&1 | FileCheck %s

ssavc4.module @vdw_store_vpm_invalid {
  ssavc4.func @bad_dynamic_multirow_kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %y = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %active = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    // CHECK: supports dynamic active_lanes only for single-row VDW stores
    ssavc4.vdw.store_vpm %addr, %y, %x, %active {elem_bytes = 4 : i32, row_len = 4 : i32, nrows = 2 : i32, memory_pitch_bytes = 16 : i32, orientation = "horizontal", serialize = "mutex"} : i32, i32, i32, i32
    ssavc4.thread_end
  }

  ssavc4.func @bad_active_lanes_attr_kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %y = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %x = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    // CHECK: active_lanes must match row_len for VDW stores
    ssavc4.vdw.store_vpm %addr, %y, %x {elem_bytes = 4 : i32, row_len = 4 : i32, nrows = 1 : i32, memory_pitch_bytes = 16 : i32, active_lanes = 16 : i32, orientation = "horizontal", serialize = "mutex"} : i32, i32, i32
    ssavc4.thread_end
  }
}
