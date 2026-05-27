// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: ssavc4.module @vdr_load_roundtrip
// CHECK: ssavc4.vdr.load
// CHECK-SAME: elem_bytes = 4 : i32
// CHECK-SAME: memory_pitch_bytes = 64 : i32
// CHECK-SAME: nrows = 16 : i32
// CHECK-SAME: orientation = "horizontal"
// CHECK-SAME: row_len = 16 : i32
// CHECK-SAME: serialize = "mutex"
// CHECK-SAME: vpitch = 1 : i32
// CHECK-SAME: vpm_base_col = 0 : i32
// CHECK-SAME: vpm_base_row = 0 : i32
ssavc4.module @vdr_load_roundtrip {
  ssavc4.func @vdr_load_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      uses_barrier = true,
      uses_shared_vpm = true,
      shared_vpm_bytes = 1024 : i32,
      require_full_block_residency = true,
      semaphores_per_block = 4 : i32,
      warps_per_block_max = 4 : i32
    }
  } {
    %addr = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    ssavc4.vdr.load %addr {
      elem_bytes = 4 : i32,
      row_len = 16 : i32,
      nrows = 16 : i32,
      memory_pitch_bytes = 64 : i32,
      vpm_base_row = 0 : i32,
      vpm_base_col = 0 : i32,
      orientation = "horizontal",
      vpitch = 1 : i32,
      serialize = "mutex"
    } : i32
    ssavc4.thread_end
  }
}
