// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @cooperative_barrier_lowering
// CHECK: vc4.func @kernel
// CHECK-SAME: schedule_mode = "cooperative_block"
// CHECK-SAME: uses_barrier = true
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
ssavc4.module @cooperative_barrier_lowering {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "cooperative_barrier_lowering",
      code_symbol = "cooperative_barrier_lowering_shader",
      tail_policy = "tail_safe",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 1 : i32}
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
    ssavc4.barrier {
      arrive_offset = 0 : i32,
      go_offset = 1 : i32,
      depart_offset = 2 : i32,
      reset_offset = 3 : i32
    }
    ssavc4.thread_end
  }
}
