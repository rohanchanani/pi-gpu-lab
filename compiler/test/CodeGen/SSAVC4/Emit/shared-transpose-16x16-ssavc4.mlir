// RUN: rm -rf %t.lowered.mlir %t.bundle
// RUN: vc4-opt %s --convert-ssavc4-to-vc4 -o %t.lowered.mlir
// RUN: FileCheck %s --input-file=%t.lowered.mlir
// RUN: vc4-codegen %t.lowered.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/manifest.json
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/kernels/shared_transpose_16x16_ssavc4.qasm

// CHECK: public_name = "shared_transpose_16x16_ssavc4"
// CHECK: tail_policy = "exact_multiple"
// CHECK: schedule_mode = "cooperative_block"
// CHECK: uses_barrier = true
// CHECK: uses_shared_vpm = true
// CHECK: vc4.qpu.sema <release>
// CHECK: waddr_add = 48 : i32
// CHECK: raddr_a = 48 : i32
// CHECK: sig = #vc4.qpu_signal<thrend>
ssavc4.module @shared_transpose_16x16_ssavc4 {
  ssavc4.func @shared_transpose_16x16_ssavc4_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    lowering_template = "shared_transpose_16x16_f32_tail_safe",
    "vc4.launch_abi" = {
      public_name = "shared_transpose_16x16_ssavc4",
      code_symbol = "shared_transpose_16x16_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 7 : i32,
      args = [
        {name = "input", kind = "buffer", direction = "in", elem_type = "u32", uniform_index = 0 : i32},
        {name = "output", kind = "buffer", direction = "out", elem_type = "u32", uniform_index = 1 : i32},
        {name = "logical_warp_id", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 2 : i32},
        {name = "warps_per_block", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 3 : i32},
        {name = "vpm_base_row", kind = "scalar", direction = "by_value", type = "u32", uniform_index = 4 : i32}
      ],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 5 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 6 : i32}
      ]
    },
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
    %tile = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    ssavc4.vpm.write %row0, %tile {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal"} : i32, vector<16xi32>
    ssavc4.barrier {arrive_offset = 0 : i32, go_offset = 1 : i32, depart_offset = 2 : i32, reset_offset = 3 : i32}
    %read = ssavc4.vpm.read %row0 {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "vertical"} : i32 -> vector<16xi32>
    ssavc4.vpm.write %row0, %read {elem_bytes = 4 : i32, lanes = 16 : i32, orientation = "horizontal"} : i32, vector<16xi32>
    ssavc4.thread_end
  }
}
