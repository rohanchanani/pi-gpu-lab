// RUN: not vc4-opt %s --verify-diagnostics

vc4.module @add_arity_error {
  vc4.func @bad_add(%a: i32) attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.alu.add <add> %a {cond = #vc4.cond<always>} : (i32) -> i32 // expected-error {{expects exactly 2 operands for this opcode}}
    vc4.return
  }
}

vc4.module @add_type_error {
  vc4.func @bad_fadd(%a: i32, %b: i32) attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.alu.add <fadd> %a, %b {cond = #vc4.cond<always>} : (i32, i32) -> i32 // expected-error {{requires f32 or vector<16xf32> types}}
    vc4.return
  }
}

vc4.module @conversion_shape_error {
  vc4.func @bad_ftoi(%a: vector<16xf32>) attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.alu.add <ftoi> %a {cond = #vc4.cond<always>} : (vector<16xf32>) -> i32 // expected-error {{operand and result must have compatible scalar or 16-lane vector shapes}}
    vc4.return
  }
}

vc4.module @mul_nop_error {
  vc4.func @bad_mul(%a: i32, %b: i32) attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.alu.mul <nop> %a, %b {cond = #vc4.cond<always>} : (i32, i32) -> i32 // expected-error {{structured vc4.alu.mul does not support the nop opcode}}
    vc4.return
  }
}

vc4.module @load_imm_missing_value {
  vc4.func @bad_ldi0() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.load_imm {mode = #vc4.load_imm_mode<splat32>} : i32 // expected-error {{requires a 'value' attribute}}
    vc4.return
  }
}

vc4.module @load_imm_splat_payload_error {
  vc4.func @bad_ldi1() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.load_imm {mode = #vc4.load_imm_mode<splat32>, value = array<i32: 1, 2, 3, 4>} : i32 // expected-error {{splat32 mode requires a signless i32 'value' attribute}}
    vc4.return
  }
}

vc4.module @load_imm_lane_count_error {
  vc4.func @bad_ldi2() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.load_imm {mode = #vc4.load_imm_mode<per_elem_u2>, value = array<i32: 0, 1, 2, 3>} : vector<16xi32> // expected-error {{per-element mode requires exactly 16 lane values}}
    vc4.return
  }
}

vc4.module @load_imm_lane_range_error {
  vc4.func @bad_ldi3() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.load_imm {mode = #vc4.load_imm_mode<per_elem_i2>, value = array<i32: -2, -1, 0, 1, 2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1>} : vector<16xi32> // expected-error {{lane values for mode per_elem_i2 must be in range \[-2, 1\]}}
    vc4.return
  }
}

vc4.module @load_imm_result_type_error {
  vc4.func @bad_ldi4() attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = vc4.load_imm {mode = #vc4.load_imm_mode<per_elem_u2>, value = array<i32: 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3>} : vector<16xf32> // expected-error {{per-element mode result type must be vector<16xi32>}}
    vc4.return
  }
}
