// RUN: vc4-opt %s --convert-ssavc4-to-vc4 | FileCheck %s

// CHECK-LABEL: vc4.module @cond_select_rematerialize_flags_ssavc4
// CHECK: vc4.func @cond_select_rematerialize_flags_kernel
// CHECK: set_flags
// CHECK-NOT: set_flags
// CHECK: cond_add = #vc4.cond<cs>
// CHECK: set_flags
// CHECK-NOT: set_flags
// CHECK: cond_add = #vc4.cond<cs>
// CHECK-NOT: ssavc4.
ssavc4.module @cond_select_rematerialize_flags_ssavc4 {
  ssavc4.func @cond_select_rematerialize_flags_kernel() attributes {
    kernel,
    threading = #vc4.threading_mode<single>,
    "vc4.launch_abi" = {
      public_name = "cond_select_rematerialize_flags_ssavc4",
      tail_policy = "exact_multiple",
      uniform_words_per_qpu = 0 : i32,
      args = [],
      builtins = []
    }
  } {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
    %all = ssavc4.load_imm <splat32> {value = -1 : i32} : vector<16xi32>
    %rows = ssavc4.load_imm <per_elem_u2> {values = array<i32: 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3>} : vector<16xi32>
    %cols = ssavc4.load_imm <per_elem_u2> {values = array<i32: 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3>} : vector<16xi32>
    %active_rows = ssavc4.load_imm <splat32> {value = 3 : i32} : vector<16xi32>
    %active_cols = ssavc4.load_imm <splat32> {value = 2 : i32} : vector<16xi32>
    %row_flags = ssavc4.make_flags %rows, %active_rows {kind = #ssavc4.flag_kind<compare>} : (vector<16xi32>, vector<16xi32>) -> !ssavc4.flags
    %col_flags = ssavc4.make_flags %cols, %active_cols {kind = #ssavc4.flag_kind<compare>} : (vector<16xi32>, vector<16xi32>) -> !ssavc4.flags
    %row_mask = ssavc4.cond_select %row_flags, %all, %zero {cond = #vc4.cond<cs>} : !ssavc4.flags, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %col_mask = ssavc4.cond_select %col_flags, %all, %zero {cond = #vc4.cond<cs>} : !ssavc4.flags, vector<16xi32>, vector<16xi32> -> vector<16xi32>
    %combined = ssavc4.alu.add %row_mask, %col_mask {opcode = #vc4.add_opcode<and>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    %sink = ssavc4.alu.add %combined, %zero {opcode = #vc4.add_opcode<or>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    ssavc4.thread_end
  }
}
