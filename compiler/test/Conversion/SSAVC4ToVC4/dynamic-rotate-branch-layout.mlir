// RUN: vc4-opt %s --convert-ssavc4-to-vc4 --vc4-verify-scheduled-hardware-rules | FileCheck %s

// CHECK-LABEL: vc4.module @dynamic_rotate_branch_layout
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<any_z_set>
// CHECK: op_add = #vc4.add_opcode<and>
// CHECK-SAME: small_imm = 15 : i32
// CHECK-SAME: waddr_add = 37 : i32
// CHECK-NEXT: vc4.qpu.bundle
// CHECK-SAME: cond_add = #vc4.cond<never>
// CHECK-SAME: op_add = #vc4.add_opcode<nop>
// CHECK: waddr_add = 34 : i32
// CHECK: small_imm = 48 : i32
// CHECK: vc4.qpu.branch attributes {cond = #vc4.branch_cond<always>
ssavc4.module @dynamic_rotate_branch_layout {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %flags = ssavc4.make_flags %one, %zero {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    ssavc4.cond_br %flags, ^skip, ^rotate_path {cond = #vc4.branch_cond<any_z_set>} : !ssavc4.flags
  ^rotate_path:
    %amount = ssavc4.load_imm <splat32> {value = 31 : i32} : i32
    %v = ssavc4.element_number : vector<16xi32>
    %r = ssavc4.rotate %v, %amount : vector<16xi32>, i32 -> vector<16xi32>
    ssavc4.br ^done
  ^skip:
    %fallback = ssavc4.load_imm <splat32> {value = 2 : i32} : vector<16xi32>
    ssavc4.br ^done
  ^done:
    ssavc4.thread_end
  }
}
