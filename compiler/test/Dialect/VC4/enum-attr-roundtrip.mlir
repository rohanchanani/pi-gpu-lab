// RUN: vc4-opt %s | FileCheck %s

module attributes {
  vc4.add = #vc4.add_opcode<add>,
  vc4.add_sat = #vc4.add_opcode<v8adds>,
  vc4.branch = #vc4.branch_cond<always>,
  vc4.cond = #vc4.cond<zs>,
  vc4.imm_splat = #vc4.load_imm_mode<splat32>,
  vc4.imm_i2 = #vc4.load_imm_mode<per_elem_i2>,
  vc4.imm_u2 = #vc4.load_imm_mode<per_elem_u2>,
  vc4.mul = #vc4.mul_opcode<fmul>,
  vc4.mul_pack = #vc4.mul_pack_mode<to_8c>,
  vc4.pack = #vc4.regfile_a_pack_mode<sat8d>,
  vc4.sig = #vc4.qpu_signal<last_thread_switch>,
  vc4.mux = #vc4.qpu_mux<b>,
  vc4.r4_unpack = #vc4.r4_unpack_mode<replicate_8d>,
  vc4.unpack = #vc4.regfile_a_unpack_mode<color8c>,
  vc4.vpmvcd_side = #vc4.vpmvcd_side<read>
} {
}

// CHECK: module attributes {
// CHECK-SAME: vc4.add = #vc4.add_opcode<add>
// CHECK-SAME: vc4.add_sat = #vc4.add_opcode<v8adds>
// CHECK-SAME: vc4.branch = #vc4.branch_cond<always>
// CHECK-SAME: vc4.cond = #vc4.cond<zs>
// CHECK-SAME: vc4.imm_i2 = #vc4.load_imm_mode<per_elem_i2>
// CHECK-SAME: vc4.imm_splat = #vc4.load_imm_mode<splat32>
// CHECK-SAME: vc4.imm_u2 = #vc4.load_imm_mode<per_elem_u2>
// CHECK-SAME: vc4.mul = #vc4.mul_opcode<fmul>
// CHECK-SAME: vc4.mul_pack = #vc4.mul_pack_mode<to_8c>
// CHECK-SAME: vc4.mux = #vc4.qpu_mux<b>
// CHECK-SAME: vc4.pack = #vc4.regfile_a_pack_mode<sat8d>
// CHECK-SAME: vc4.r4_unpack = #vc4.r4_unpack_mode<replicate_8d>
// CHECK-SAME: vc4.sig = #vc4.qpu_signal<last_thread_switch>
// CHECK-SAME: vc4.unpack = #vc4.regfile_a_unpack_mode<color8c>
// CHECK-SAME: vc4.vpmvcd_side = #vc4.vpmvcd_side<read>
