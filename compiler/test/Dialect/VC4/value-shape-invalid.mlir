// RUN: vc4-opt %s --verify-diagnostics

vc4.module @pack_mode_selection_error {
  vc4.func @bad_pack(%arg0: i32) attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = "vc4.pack"(%arg0) : (i32) -> i32 // expected-error {{requires exactly one of 'regfile_a_mode' or 'mul_mode'}}
    vc4.return
  }
}

vc4.module @pack_mul_type_error {
  vc4.func @bad_pack_mul(%arg0: i32) attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = "vc4.pack"(%arg0) <{mul_mode = #vc4.mul_pack_mode<to_8a>}> : (i32) -> i32 // expected-error {{mul_mode requires f32 or vector<16xf32> input}}
    vc4.return
  }
}

vc4.module @unpack_mode_selection_error {
  vc4.func @bad_unpack(%arg0: i32) attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = "vc4.unpack"(%arg0) <{regfile_a_mode = #vc4.regfile_a_unpack_mode<none>}> : (i32) -> i32 // expected-error {{regfile_a_mode must not be <none>}}
    vc4.return
  }
}

vc4.module @unpack_r4_type_error {
  vc4.func @bad_unpack_r4(%arg0: i32) attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = "vc4.unpack"(%arg0) <{r4_mode = #vc4.r4_unpack_mode<color8a>}> : (i32) -> i32 // expected-error {{r4_mode requires f32 or vector<16xf32> result}}
    vc4.return
  }
}

vc4.module @rotate_exclusive_error {
  vc4.func @bad_rotate_both(%arg0: vector<16xi32>, %amt: i32) attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = "vc4.rotate"(%arg0, %amt) <{immediate = 3 : i32}> : (vector<16xi32>, i32) -> vector<16xi32> // expected-error {{requires exactly one of an amount operand or an immediate attribute}}
    vc4.return
  }
}

vc4.module @rotate_missing_amount_error {
  vc4.func @bad_rotate_none(%arg0: vector<16xi32>) attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = "vc4.rotate"(%arg0) : (vector<16xi32>) -> vector<16xi32> // expected-error {{requires exactly one of an amount operand or an immediate attribute}}
    vc4.return
  }
}

vc4.module @rotate_shape_error {
  vc4.func @bad_rotate_shape(%arg0: vector<8xi32>) attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = "vc4.rotate"(%arg0) <{immediate = 1 : i32}> : (vector<8xi32>) -> vector<8xi32> // expected-error {{input and result type must be vector<16xi32> or vector<16xf32>}}
    vc4.return
  }
}

vc4.module @rotate_immediate_range_error {
  vc4.func @bad_rotate_imm(%arg0: vector<16xf32>) attributes {threading = 0 : i32, form = 0 : i32} {
    %0 = "vc4.rotate"(%arg0) <{immediate = 16 : i32}> : (vector<16xf32>) -> vector<16xf32> // expected-error {{immediate rotate amount must be in range}}
    vc4.return
  }
}
