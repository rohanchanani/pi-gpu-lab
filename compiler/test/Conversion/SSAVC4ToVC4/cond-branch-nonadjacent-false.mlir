// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @cond_branch_nonadjacent_false
// CHECK: vc4.func @kernel
// CHECK: vc4.qpu.branch attributes {{.*}}cond = #vc4.branch_cond<any_c_set>
// CHECK: vc4.qpu.branch attributes {{.*}}cond = #vc4.branch_cond<always>
// CHECK: vc4.qpu.branch attributes {{.*}}cond = #vc4.branch_cond<always>
// CHECK-COUNT-1: sig = #vc4.qpu_signal<thrend>
ssavc4.module @cond_branch_nonadjacent_false {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %entry_flags = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %entry_flags, ^early, ^compute {cond = #vc4.branch_cond<any_z_set>} : !ssavc4.flags
  ^compute:
    %sum = ssavc4.alu.add %zero, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^tail_gate
  ^tail_gate:
    %tail_flags = ssavc4.make_flags %sum, %zero {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %tail_flags, ^done, ^tail {cond = #vc4.branch_cond<any_c_set>} : !ssavc4.flags
  ^early:
    ssavc4.thread_end
  ^tail:
    %two = ssavc4.alu.add %sum, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.br ^done
  ^done:
    ssavc4.thread_end
  }
}
