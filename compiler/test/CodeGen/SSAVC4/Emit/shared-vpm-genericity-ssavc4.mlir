// RUN: rm -rf %t.lowered.mlir %t.bundle
// RUN: vc4-opt %s --convert-ssavc4-to-vc4 -o %t.lowered.mlir
// RUN: FileCheck %s --input-file=%t.lowered.mlir
// RUN: vc4-codegen %t.lowered.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/manifest.json
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/kernels/neutral_shared_vpm_entry.qasm

// CHECK: public_name = "neutral_shared_vpm_entry"
// CHECK: schedule_mode = "cooperative_block"
// CHECK: uses_shared_vpm = true
// CHECK: value = 1055232 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: waddr_add = 48 : i32
// CHECK: value = 1053184 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<read>
// CHECK: raddr_a = 48 : i32
// CHECK: vc4.qpu.vpmvcd_setup
// CHECK-SAME: side = #vc4.vpmvcd_side<write>
// CHECK: waddr_add = 48 : i32
// CHECK: sig = #vc4.qpu_signal<thrend>
ssavc4.module @neutral_shared_vpm_codegen {
  ssavc4.func @neutral_shared_vpm_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "neutral_shared_vpm_entry",
      code_symbol = "neutral_shared_vpm_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    },
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
