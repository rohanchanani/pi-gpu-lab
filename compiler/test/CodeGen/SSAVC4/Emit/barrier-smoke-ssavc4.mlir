// RUN: rm -rf %t.lowered.mlir %t.bundle
// RUN: vc4-opt %s --convert-ssavc4-to-vc4 -o %t.lowered.mlir
// RUN: vc4-opt %t.lowered.mlir --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o /dev/null
// RUN: vc4-codegen %t.lowered.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernels/qpu_barrier_syncthreads_ssavc4.qasm
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=QASM --input-file=%t.bundle/kernels/qpu_barrier_syncthreads_ssavc4.qasm
// RUN: FileCheck %s --check-prefix=MANIFEST --input-file=%t.bundle/manifest.json

// QASM: srel
// QASM: sacq
// QASM: thrend
// MANIFEST: "public_name": "qpu_barrier_syncthreads_ssavc4"
// MANIFEST: "code_symbol": "qpu_barrier_syncthreads_ssavc4_shader"
// MANIFEST: "schedule_mode": "cooperative_block"
// MANIFEST: "uses_barrier": true
// MANIFEST: "require_full_block_residency": true
// MANIFEST: "semaphores_per_block": 4
ssavc4.module @qpu_barrier_syncthreads_ssavc4_codegen {
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
    ssavc4.barrier {arrive_offset = 0 : i32, go_offset = 1 : i32, depart_offset = 2 : i32, reset_offset = 3 : i32}
    ssavc4.thread_end
  }
}
