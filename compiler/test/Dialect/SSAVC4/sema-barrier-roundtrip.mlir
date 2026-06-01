// RUN: vc4-opt %s | FileCheck %s

// CHECK-LABEL: ssavc4.module @sema_barrier_roundtrip
// CHECK: ssavc4.sema.release
// CHECK-SAME: id = 0 : i32
// CHECK: ssavc4.sema.acquire
// CHECK-SAME: id = 1 : i32
// CHECK: ssavc4.barrier
// CHECK-SAME: arrive_offset = 0 : i32
// CHECK-SAME: depart_offset = 2 : i32
// CHECK-SAME: go_offset = 1 : i32
// CHECK-SAME: reset_offset = 3 : i32
ssavc4.module @sema_barrier_roundtrip {
  ssavc4.func @sema_barrier_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block = 12 : i32,
      user_vpm_rows_per_block = 0 : i32,
      compiler_vpm_staging_rows_per_warp = 0 : i32,
      compiler_vpm_staging_rows_per_block = 0 : i32,
      total_vpm_rows_per_block = 0 : i32,
      uses_tmu = false,
      uses_vpm = false,
      uses_vpm_qpu_read = false,
      uses_vpm_qpu_write = false,
      uses_vdr = false,
      uses_vdw = false,
      uses_barrier = true,
      semaphore_count_per_block = 4 : i32,
      requires_vpm_base_row_builtin = false,
      requires_semaphore_base_builtin = true
    }
  } {
    %s0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %s1 = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    ssavc4.sema.release %s0 {id = 0 : i32} : i32
    ssavc4.sema.acquire %s1 {id = 1 : i32} : i32
    ssavc4.barrier {arrive_offset = 0 : i32, go_offset = 1 : i32, depart_offset = 2 : i32, reset_offset = 3 : i32}
    ssavc4.thread_end
  }
}
