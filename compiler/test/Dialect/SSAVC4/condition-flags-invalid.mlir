// RUN: vc4-opt %s -split-input-file -verify-diagnostics

module {
  %x = ssavc4.load_imm <splat32> {value = 1.000000e+00 : f32} : f32
  // expected-error@+1 {{operand type must be an SSAVC4 integer carrier}}
  %flags = ssavc4.make_flags %x {kind = #ssavc4.flag_kind<zero_test>} : (f32) -> !ssavc4.flags
}

// ----

module {
  %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
  %v = ssavc4.load_imm <splat32> {value = 1 : i32} : vector<16xi32>
  // expected-error@+1 {{operand and operand must have the same SSAVC4 scalar/vector shape}}
  %flags = ssavc4.make_flags %x, %v {kind = #ssavc4.flag_kind<sub>} : (i32, vector<16xi32>) -> !ssavc4.flags
}

// ----

module {
  %mask = ssavc4.load_imm <splat32> {value = 1 : i32} : vector<16xi32>
  %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
  // expected-error@+1 {{flags values must have at most one use}}
  %flags = ssavc4.make_flags %mask {kind = #ssavc4.flag_kind<zero_test>} : (vector<16xi32>) -> !ssavc4.flags
  %a = ssavc4.cond_select %flags, %mask, %zero {cond = #vc4.cond<zc>} : !ssavc4.flags, vector<16xi32>, vector<16xi32> -> vector<16xi32>
  %b = ssavc4.cond_select %flags, %zero, %mask {cond = #vc4.cond<zc>} : !ssavc4.flags, vector<16xi32>, vector<16xi32> -> vector<16xi32>
}

// ----

module {
  %mask = ssavc4.load_imm <splat32> {value = 1 : i32} : vector<16xi32>
  %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : vector<16xi32>
  %flags = ssavc4.make_flags %mask {kind = #ssavc4.flag_kind<zero_test>} : (vector<16xi32>) -> !ssavc4.flags
  // expected-error@+1 {{cond_select requires a real per-lane condition}}
  %bad = ssavc4.cond_select %flags, %mask, %zero {cond = #vc4.cond<always>} : !ssavc4.flags, vector<16xi32>, vector<16xi32> -> vector<16xi32>
}

// ----

ssavc4.module @cond_br_always_invalid {
  ssavc4.func @kernel() attributes {kernel, threading = #vc4.threading_mode<single>} {
    %zero = ssavc4.load_imm <splat32> {value = 0 : i32} : i32
    %one = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
    %flags = ssavc4.make_flags %zero, %one {kind = #ssavc4.flag_kind<sub>} : (i32, i32) -> !ssavc4.flags
    // expected-error@+1 {{cond_br with always condition is invalid; use ssavc4.br}}
    ssavc4.cond_br %flags, ^done, ^done {cond = #vc4.branch_cond<always>} : !ssavc4.flags
  ^done:
    ssavc4.thread_end
  }
}
