// RUN: vc4-opt %s --convert-ssavc4-to-vc4 --vc4-verify-scheduled-adjacent-hazards | FileCheck %s

// CHECK-LABEL: vc4.module @block_args_forward_merge_vector
// CHECK-NOT: ssavc4.phi
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<any_c_clear>
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<always>
// CHECK-SAME: raddr_a = 14 : i32
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// CHECK: vc4.qpu.branch attributes
// CHECK-SAME: cond = #vc4.branch_cond<always>
// CHECK-SAME: raddr_a = 14 : i32
ssavc4.module @block_args_forward_merge_vector {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %left = ssavc4.load_imm <splat32> {value = 1065353216 : i32} : vector<16xf32>
    %right = ssavc4.load_imm <splat32> {value = 1073741824 : i32} : vector<16xf32>
    %flags = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %flags, ^merge(%zero, %left : i32, vector<16xf32>), ^merge(%one, %right : i32, vector<16xf32>) {cond = #vc4.branch_cond<any_c_clear>} : !ssavc4.flags

  ^merge(%tag: i32, %value: vector<16xf32>):
    %sink = ssavc4.alu.add %value, %left {opcode = #vc4.add_opcode<fadd>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    %tag_sink = ssavc4.alu.add %tag, %one {opcode = #vc4.add_opcode<add>} : (i32, i32) -> i32
    ssavc4.thread_end
  }
}
