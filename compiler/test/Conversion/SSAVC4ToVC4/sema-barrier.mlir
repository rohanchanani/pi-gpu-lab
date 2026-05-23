// RUN: vc4-opt %s --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses | FileCheck %s

// CHECK-LABEL: vc4.module @sema_barrier_lowering
// CHECK: vc4.resource =
// CHECK: schedule_mode = "cooperative_block"
// CHECK: uses_barrier = true
// CHECK: vc4.qpu.sema <release>
// CHECK-SAME: id = 0 : i32
// CHECK: vc4.qpu.sema <acquire>
// CHECK-SAME: id = 1 : i32
// CHECK: vc4.qpu.sema <release>
// CHECK-SAME: id = 0 : i32
// CHECK: vc4.qpu.sema <acquire>
// CHECK-SAME: id = 0 : i32
// CHECK: vc4.qpu.sema <release>
// CHECK-SAME: id = 1 : i32
// CHECK: vc4.qpu.sema <acquire>
// CHECK-SAME: id = 1 : i32
// CHECK: vc4.qpu.sema <release>
// CHECK-SAME: id = 2 : i32
// CHECK: vc4.qpu.sema <acquire>
// CHECK-SAME: id = 2 : i32
// CHECK: vc4.qpu.sema <release>
// CHECK-SAME: id = 3 : i32
// CHECK: vc4.qpu.sema <acquire>
// CHECK-SAME: id = 3 : i32
// CHECK: sig = #vc4.qpu_signal<thrend>
// CHECK-NOT: ssavc4.
ssavc4.module @sema_barrier_lowering {
  ssavc4.func @sema_barrier_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "sema_barrier_lowering",
      code_symbol = "sema_barrier_lowering_shader",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "logical_warp_id", kind = #vc4.builtin_kind<logical_warp_id>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "warps_per_block", kind = #vc4.builtin_kind<warps_per_block>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    },
    "vc4.resource" = {
      schedule_mode = "cooperative_block",
      warps_per_block_max = 12 : i32,
      uses_shared_vpm = false,
      uses_barrier = true,
      semaphores_per_block = 4 : i32,
      require_full_block_residency = true
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
