// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  // expected-error @+1 {{vc4value.kernel requires integer attr vc4value.grid_rank in [1, 3]}}
  func.func @missing_grid_rank() attributes {vc4value.kernel} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{vc4value.kernel requires integer attr vc4value.grid_rank in [1, 3]}}
  func.func @grid_rank_zero() attributes {vc4value.kernel, vc4value.grid_rank = 0 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{vc4value.kernel requires integer attr vc4value.grid_rank in [1, 3]}}
  func.func @grid_rank_four() attributes {vc4value.kernel, vc4value.grid_rank = 4 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @missing_arg_name argument #0: requires StringAttr vc4value.arg_name matching ^[A-Za-z_][A-Za-z0-9_]*$}}
  func.func @missing_arg_name(%x: i32) attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @duplicate_arg_name argument #1 'x': duplicate vc4value.arg_name 'x'}}
  func.func @duplicate_arg_name(
      %a: i32 {vc4value.arg_name = "x"},
      %b: f32 {vc4value.arg_name = "x"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @bad_arg_name argument #0 '1bad': vc4value.arg_name must match ^[A-Za-z_][A-Za-z0-9_]*$}}
  func.func @bad_arg_name(%x: i32 {vc4value.arg_name = "1bad"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @reserved_arg_name argument #0 'program_id': vc4value.arg_name 'program_id' is reserved for lower-half/runtime/builtin metadata}}
  func.func @reserved_arg_name(%x: i32 {vc4value.arg_name = "program_id"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @missing_direction argument #0 'x': memref argument requires vc4value.direction = "in", "out", or "inout"}}
  func.func @missing_direction(%x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @bad_direction argument #0 'x': vc4value.direction must be "in", "out", or "inout"}}
  func.func @bad_direction(%x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "read"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @scalar_direction argument #0 'x': non-memref scalar argument must not have vc4value.direction}}
  func.func @scalar_direction(%x: i32 {vc4value.arg_name = "x", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @bad_scalar_role argument #0 'n': vc4value.scalar_role must be "value", "extent", "stride", "grid_dim", or "policy"}}
  func.func @bad_scalar_role(%n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extentish"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @vector_arg argument #0 'v': vector public arguments are not legal in the VC4 value ABI}}
  func.func @vector_arg(%v: vector<16xf32> {vc4value.arg_name = "v"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  // expected-error @+2 {{tensor types are not legal in the initial VC4 value surface}}
  // expected-error @+1 {{public value kernel @tensor_arg argument #0 't': tensor public arguments are not legal in the VC4 value ABI}}
  func.func @tensor_arg(%t: tensor<16xf32> {vc4value.arg_name = "t"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
  func.func @helper_program_id() {
    // expected-error @+1 {{vc4value launch identity ops require an enclosing vc4value.kernel function}}
    %pid = vc4value.program_id {axis = 0 : i32} : index
    return
  }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @unknown_vc4value_arg_attr argument #0 'x': unknown vc4value public argument ABI attribute 'vc4value.typo'}}
  func.func @unknown_vc4value_arg_attr(%x: i32 {vc4value.arg_name = "x", vc4value.typo = "bad"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    return
  }
}
