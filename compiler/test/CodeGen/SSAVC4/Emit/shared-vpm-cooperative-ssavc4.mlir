// RUN: rm -rf %t.lowered.mlir %t.bundle
// RUN: vc4-opt %s --convert-ssavc4-to-vc4 -o %t.lowered.mlir
// RUN: FileCheck %s --input-file=%t.lowered.mlir
// RUN: vc4-codegen %t.lowered.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/manifest.json
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/kernels/shared_transpose_16x16_ssavc4.qasm

// CHECK: public_name = "shared_transpose_16x16_ssavc4"
// CHECK: schedule_mode = "cooperative_block"
// CHECK: uses_shared_vpm = true
// CHECK: value = 1055232 : i32
// CHECK: waddr_add = 48 : i32
// CHECK: vc4.qpu.sema <release>
// CHECK: value = 1055232 : i32
// CHECK: raddr_a = 48 : i32
// CHECK: sig = #vc4.qpu_signal<thrend>
ssavc4.module @shared_transpose_16x16_ssavc4 {
  ssavc4.func @shared_transpose_16x16_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "shared_transpose_16x16_ssavc4",
      code_symbol = "shared_transpose_16x16_ssavc4_shader",
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
      uses_barrier = true,
      uses_shared_vpm = true,
      shared_vpm_bytes = 1024 : i32,
      require_full_block_residency = true,
      warps_per_block_max = 2 : i32,
      semaphores_per_block = 4 : i32
    }
  } {
    %row0 = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %row1 = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %seed = ssavc4.load_imm <splat32> {value = 42 : i32} : vector<16xi32>
    ssavc4.vpm.write %row0, %seed {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal"} : i32, vector<16xi32>
    ssavc4.barrier {arrive_offset = 0 : i32, go_offset = 1 : i32, depart_offset = 2 : i32, reset_offset = 3 : i32}
    %roundtrip = ssavc4.vpm.read %row0 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal"} : i32 -> vector<16xi32>
    ssavc4.vpm.write %row1, %roundtrip {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal"} : i32, vector<16xi32>
    ssavc4.barrier {arrive_offset = 0 : i32, go_offset = 1 : i32, depart_offset = 2 : i32, reset_offset = 3 : i32}
    ssavc4.thread_end
  }
}
