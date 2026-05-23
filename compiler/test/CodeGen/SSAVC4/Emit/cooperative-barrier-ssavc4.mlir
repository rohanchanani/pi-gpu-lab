// RUN: rm -rf %t.lowered.mlir %t.bundle
// RUN: vc4-opt %s --convert-ssavc4-to-vc4 -o %t.lowered.mlir
// RUN: FileCheck %s --input-file=%t.lowered.mlir
// RUN: vc4-codegen %t.lowered.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/manifest.json
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/kernels/qpu_barrier_syncthreads_ssavc4.qasm
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json

// CHECK-LABEL: vc4.module @qpu_barrier_syncthreads_ssavc4
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
// MANIFEST-DAG: "schedule_mode": "cooperative_block"
// MANIFEST-DAG: "uses_barrier": true
// MANIFEST-DAG: "require_full_block_residency": true
// MANIFEST-DAG: "warps_per_block_max": 12
// MANIFEST-DAG: "semaphores_per_block": 4
ssavc4.module @qpu_barrier_syncthreads_ssavc4 {
  ssavc4.func @qpu_barrier_syncthreads_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "qpu_barrier_syncthreads_ssavc4",
      code_symbol = "qpu_barrier_syncthreads_ssavc4_shader",
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
    ssavc4.barrier {
      arrive_offset = 0 : i32,
      go_offset = 1 : i32,
      depart_offset = 2 : i32,
      reset_offset = 3 : i32
    }
    ssavc4.thread_end
  }
}
