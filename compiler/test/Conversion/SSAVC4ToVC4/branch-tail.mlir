// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @branch_tail_ssavc4
// CHECK: vc4.func @branch_tail_kernel
// CHECK: vc4.qpu.bundle
// CHECK-SAME: op_add = #vc4.add_opcode<sub>
// CHECK-SAME: set_flags
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<any_c_set>
// CHECK-SAME: immediate = 72 : i32
// CHECK-SAME: relative = true
// CHECK-SAME: use_reg = false
// CHECK-SAME: waddr_add = 31 : i32
// CHECK-SAME: waddr_mul = 30 : i32
// CHECK: {
// CHECK: vc4.qpu.bundle
// CHECK: vc4.qpu.bundle
// CHECK: vc4.qpu.bundle
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<always>
// CHECK-SAME: immediate = 32 : i32
// CHECK: sig = #vc4.qpu_signal<thrend>
ssavc4.module @branch_tail_ssavc4 {
  ssavc4.func @branch_tail_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "branch_tail_ssavc4",
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
    %flags = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %flags, ^done, ^body {cond = #vc4.branch_cond<any_c_set>} : !ssavc4.flags
  ^body:
    %two = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
    ssavc4.br ^done
  ^done:
    ssavc4.thread_end
  }
}
