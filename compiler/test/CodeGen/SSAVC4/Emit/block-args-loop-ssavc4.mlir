// RUN: rm -rf %t.lowered.mlir %t.bundle
// RUN: vc4-opt %s --convert-ssavc4-to-vc4 -o %t.lowered.mlir
// RUN: FileCheck %s --check-prefix=LOWERED < %t.lowered.mlir
// RUN: vc4-codegen %t.lowered.mlir --emit-bundle %t.bundle
// RUN: test -f %t.bundle/kernel_launch.c
// RUN: test -f %t.bundle/kernel_launch.h
// RUN: test -f %t.bundle/kernels/block_args_loop_ssavc4.qasm
// RUN: test -f %t.bundle/manifest.json
// RUN: FileCheck %s --check-prefix=QASM < %t.bundle/kernels/block_args_loop_ssavc4.qasm

// LOWERED-NOT: ssavc4.phi
// LOWERED: vc4.qpu.branch attributes {{.*}}immediate = -{{[0-9]+}} : i32
// QASM: b
// QASM: nop
// QASM: nop
// QASM: nop

ssavc4.module @block_args_loop_ssavc4_codegen {
  ssavc4.func @kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "block_args_loop_ssavc4",
      code_symbol = "block_args_loop_ssavc4_shader",
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
    %limit = ssavc4.load_imm <splat32> {value = 3 : i32} : i32
    %acc0 = ssavc4.load_imm <splat32> {value = 7 : i32} : i32
    ssavc4.br ^loop(%zero, %acc0 : i32, i32)

  ^loop(%i: i32, %acc: i32):
    %done_flags = ssavc4.make_flags %i, %limit {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %done_flags, ^done(%acc : i32), ^body(%i, %acc : i32, i32) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^body(%i_body: i32, %acc_body: i32):
    %i_next = ssavc4.alu.add %i_body, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %acc_next = ssavc4.alu.add %acc_body, %i_next {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^loop(%i_next, %acc_next : i32, i32)

  ^done(%final: i32):
    %sink = ssavc4.alu.add %final, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.thread_end
  }
}
