// RUN: rm -rf %t.lowered.mlir %t.bundle
// RUN: vc4-opt %s --convert-ssavc4-to-vc4 -o %t.lowered.mlir
// RUN: FileCheck %s --check-prefix=LOWERED < %t.lowered.mlir
// RUN: vc4-codegen %t.lowered.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/kernels/block_args_cond_merge_ssavc4.qasm
// RUN: test -f %t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=QASM < %t.bundle/kernels/block_args_cond_merge_ssavc4.qasm

// LOWERED-NOT: ssavc4.phi
// LOWERED: vc4.qpu.branch
// LOWERED-SAME: cond = #vc4.branch_cond<any_c_clear>
// LOWERED: vc4.qpu.branch
// LOWERED-SAME: cond = #vc4.branch_cond<always>
// LOWERED: vc4.qpu.branch
// LOWERED-SAME: cond = #vc4.branch_cond<always>
// QASM: b
// QASM: nop
// QASM: nop
// QASM: nop

ssavc4.module @block_args_cond_merge_ssavc4_codegen {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "block_args_cond_merge_ssavc4",
      code_symbol = "block_args_cond_merge_ssavc4_shader",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 2 : i32,
      args = [],
      builtins = [
        {name = "qpu_id", kind = #vc4.builtin_kind<qpu_num>, materialization = "uniform_suffix", uniform_index = 0 : i32},
        {name = "num_qpus", kind = #vc4.builtin_kind<num_qpus>, materialization = "uniform_suffix", uniform_index = 1 : i32}
      ]
    }
  } {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %left = ssavc4.load_imm <splat32> {value = 11 : i32} : i32
    %right = ssavc4.load_imm <splat32> {value = 29 : i32} : i32
    %flags = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %flags, ^merge(%left : i32), ^merge(%right : i32) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags
  ^merge(%m: i32):
    %sum = ssavc4.alu.add %m, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.thread_end
  }
}
