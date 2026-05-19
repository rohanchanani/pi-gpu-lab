// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK: vc4.func @shared_vpm_kernel
// CHECK-SAME: require_full_block_residency = true
// CHECK-SAME: schedule_mode = "cooperative_block"
// CHECK-SAME: uses_shared_vpm = true
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: waddr_add = 48 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<read>
// CHECK: raddr_a = 48 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: sig = #vc4.qpu_signal<thrend>
ssavc4.module @shared_vpm_ssavc4 {
  ssavc4.func @shared_vpm_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      uses_barrier = true,
      uses_shared_vpm = true,
      shared_vpm_bytes = 1024 : i32,
      require_full_block_residency = true,
      warps_per_block_max = 4 : i32,
      semaphores_per_block = 4 : i32
    }
  } {
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %tile = ssavc4.load_imm <splat32> {value = 7 : i32} : vector<16xi32>
    ssavc4.vpm.write %row0, %tile {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal"} : i32, vector<16xi32>
    %read = ssavc4.vpm.read %row0 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "vertical"} : i32 -> vector<16xi32>
    ssavc4.vpm.write %row0, %read {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal"} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
