// RUN: vc4-opt %s --convert-ssavc4-to-vc4 --vc4-verify-scheduled-adjacent-hazards | FileCheck %s

// CHECK-LABEL: vc4.module @block_args_forward_preheader_vector
// CHECK-NOT: ssavc4.phi
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<always>
// CHECK-SAME: raddr_a = 14 : i32
ssavc4.module @block_args_forward_preheader_vector {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %limit = ssavc4.load_imm <splat32> {value = 4 : i32} : i32
    %acc0 = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xf32>
    %one_v = ssavc4.load_imm <splat32> {value = 1065353216 : i32} : vector<16xf32>
    ssavc4.br ^loop(%zero, %acc0 : i32, vector<16xf32>)

  ^loop(%i: i32, %acc: vector<16xf32>):
    %done_flags = ssavc4.make_flags %i, %limit {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %done_flags, ^exit(%acc : vector<16xf32>), ^body(%i, %acc : i32, vector<16xf32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^body(%i_body: i32, %acc_body: vector<16xf32>):
    %i_next = ssavc4.alu.add %i_body, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    %acc_next = ssavc4.alu.add %acc_body, %one_v {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    ssavc4.br ^loop(%i_next, %acc_next : i32, vector<16xf32>)

  ^exit(%final: vector<16xf32>):
    %sink = ssavc4.alu.add %final, %acc0 {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    ssavc4.thread_end
  }
}
