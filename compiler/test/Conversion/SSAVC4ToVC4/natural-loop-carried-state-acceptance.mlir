// RUN: vc4-opt %s --convert-ssavc4-to-vc4 --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards | FileCheck %s --check-prefix=LOOP
// RUN: not vc4-opt %S/block-args-irreducible-loop-unsupported.mlir --convert-ssavc4-to-vc4 2>&1 | FileCheck %s --check-prefix=IRREDUCIBLE

// LOOP-LABEL: vc4.module @natural_loop_carried_state_acceptance
// LOOP: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// LOOP: vc4.qpu.bundle {{.*}}op_add = #vc4.add_opcode<or>
// LOOP: vc4.qpu.branch attributes
// LOOP-SAME: cond = #vc4.branch_cond<always>
// LOOP-SAME: raddr_a = 14 : i32
// LOOP: vc4.qpu.branch attributes {{.*}}cond = #vc4.branch_cond<always>{{.*}}immediate = -{{[0-9]+}} : i32
// LOOP: sig = #vc4.qpu_signal<thrend>
// LOOP: vc4.qpu.bundle
// LOOP: vc4.qpu.bundle
// LOOP-NOT: vc4.qpu.
// IRREDUCIBLE: SSAVC4 block-argument lowering supports only natural loops with conservative loop-carried data values
ssavc4.module @natural_loop_carried_state_acceptance {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %limit = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
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
