// RUN: vc4-opt %s -split-input-file -verify-diagnostics

module {
  // expected-error@+1 {{result type must be one of i32, f32, vector<16xi32>, or vector<16xf32>}}
  %bad = ssavc4.load_imm <splat32> {value = 1 : i32} : i64
}

// ----

module {
  // expected-error@+1 {{result type must be vector<16xi32>}}
  %bad = ssavc4.element_number : i32
}

// ----

module {
  %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
  // expected-error@+1 {{result type must be vector<16xi32> or vector<16xf32>}}
  %bad = ssavc4.splat %x : i32 -> vector<8xi32>
}

// ----

module {
  %x = ssavc4.load_imm <splat32> {value = 1.000000e+00 : f32} : f32
  // expected-error@+1 {{input and result must have the same SSAVC4 element domain}}
  %bad = ssavc4.splat %x : f32 -> vector<16xi32>
}

// ----

module {
  %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
  // expected-error@+1 {{input and result must have the same SSAVC4 element domain}}
  %bad = ssavc4.splat %x : i32 -> vector<16xf32>
}

// ----

module {
  %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
  // expected-error@+1 {{input and result types must match}}
  %bad = ssavc4.mov %x : i32 -> vector<16xi32>
}

// ----

module {
  %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
  %y = ssavc4.load_imm <splat32> {value = 2 : i32} : i32
  // expected-error@+1 {{selected ADD opcode requires exactly one operand}}
  %bad = ssavc4.alu.add %x, %y {opcode = #vc4.add_opcode<clz>} : (i32, i32) -> i32
}

// ----

module {
  %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
  // expected-error@+1 {{requires exactly two operands}}
  %bad = ssavc4.alu.mul %x {opcode = #vc4.mul_opcode<mul24>} : (i32) -> i32
}

// ----

module {
  %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
  %v = ssavc4.splat %x : i32 -> vector<16xi32>
  // expected-error@+1 {{'amount' must be in range [0, 15]}}
  %bad = ssavc4.rotate %v {amount = 16 : i32} : vector<16xi32> -> vector<16xi32>
}

// ----

module {
  %x = ssavc4.load_imm <splat32> {value = 1 : i32} : i32
  %v = ssavc4.splat %x : i32 -> vector<16xi32>
  // expected-error@+1 {{flags values must have at most one use}}
  %flags = ssavc4.make_flags %v {kind = #ssavc4.flag_kind<zero_test>} : (vector<16xi32>) -> !ssavc4.flags
  %a = "builtin.unrealized_conversion_cast"(%flags) : (!ssavc4.flags) -> i32
  %b = "builtin.unrealized_conversion_cast"(%flags) : (!ssavc4.flags) -> i32
}
