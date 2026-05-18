// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @neutral_shared_vpm_module
// CHECK: vc4.func @neutral_shared_vpm_kernel
// CHECK: schedule_mode = "cooperative_block"
// CHECK: uses_shared_vpm = true
// CHECK: value = 1055232 : i32
// CHECK: waddr_add = 49 : i32
// CHECK: waddr_add = 48 : i32
// CHECK: value = 1053184 : i32
// CHECK: raddr_a = 48 : i32
// CHECK: waddr_add = 48 : i32
// CHECK: sig = #vc4.qpu_signal<thrend>
// CHECK-NOT: ssavc4.
ssavc4.module @neutral_shared_vpm_module {
  ssavc4.func @neutral_shared_vpm_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      uses_barrier = false,
      uses_shared_vpm = true,
      shared_vpm_bytes = 1024 : i32,
      require_full_block_residency = true,
      warps_per_block_max = 2 : i32,
      semaphores_per_block = 0 : i32
    }
  } {
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row1 = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %seed = ssavc4.load_imm <splat32> {value = 9 : i32} : vector<16xi32>
    ssavc4.vpm.write %row0, %seed {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal"} : i32, vector<16xi32>
    %read = ssavc4.vpm.read %row0 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "vertical"} : i32 -> vector<16xi32>
    ssavc4.vpm.write %row1, %read {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal"} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
