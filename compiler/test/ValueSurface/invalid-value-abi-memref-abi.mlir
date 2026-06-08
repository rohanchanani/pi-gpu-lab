// RUN: vc4-opt %s --vc4-verify-value-surface -verify-diagnostics -split-input-file

builtin.module {
  // expected-error @+1 {{public value kernel @default_memspace argument #0 'x': public memref argument memory space must be #vc4value.global}}
  func.func @default_memspace(%x: memref<16xf32> {vc4value.arg_name = "x", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @wrong_memspace argument #0 'x': public memref argument memory space must be #vc4value.global}}
  func.func @wrong_memspace(%x: memref<16xf32, 1> {vc4value.arg_name = "x", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+2 {{unranked memrefs are not legal in the VC4 value surface}}
  // expected-error @+1 {{public value kernel @unranked argument #0 'x': public memref argument must be a ranked memref with rank 1 or 2}}
  func.func @unranked(%x: memref<*xf32> {vc4value.arg_name = "x", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @rank0 argument #0 'x': public memref argument rank must be 1 or 2}}
  func.func @rank0(%x: memref<f32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+2 {{memref rank greater than 2 is not legal in the VC4 value surface}}
  // expected-error @+1 {{public value kernel @rank3 argument #0 'x': public memref argument rank must be 1 or 2}}
  func.func @rank3(%x: memref<1x1x1xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+2 {{unsupported memref element type in VC4 value surface}}
  // expected-error @+1 {{public value kernel @memref_i64 argument #0 'x': public memref element type must be signless i8, signless i16, signless i32, f16, or f32}}
  func.func @memref_i64(%x: memref<16xi64, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+2 {{unsupported memref element type in VC4 value surface}}
  // expected-error @+1 {{public value kernel @memref_f64 argument #0 'x': public memref element type must be signless i8, signless i16, signless i32, f16, or f32}}
  func.func @memref_f64(%x: memref<16xf64, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+2 {{unsupported memref element type in VC4 value surface}}
  // expected-error @+1 {{public value kernel @memref_bf16 argument #0 'x': public memref element type must be signless i8, signless i16, signless i32, f16, or f32}}
  func.func @memref_bf16(%x: memref<16xbf16, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+2 {{unsupported memref element type in VC4 value surface}}
  // expected-error @+1 {{public value kernel @memref_i1 argument #0 'x': public memref element type must be signless i8, signless i16, signless i32, f16, or f32}}
  func.func @memref_i1(%x: memref<16xi1, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @missing_shape_args argument #1 'x': dynamic public memref argument requires vc4value.shape_args}}
  func.func @missing_shape_args(
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @shape_len argument #2 'x': vc4value.shape_args length must equal the number of dynamic memref dimensions}}
  func.func @shape_len(
      %m: index {vc4value.arg_name = "m", vc4value.scalar_role = "extent"},
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %x: memref<?x?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["m"]})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @shape_missing_arg argument #1 'x': vc4value.shape_args entry 'missing' must name an existing scalar argument}}
  func.func @shape_missing_arg(
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["missing"]})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @shape_f32_arg argument #1 'x': vc4value.shape_args entry 'extent' must name an index or i32 scalar argument}}
  func.func @shape_f32_arg(
      %extent: f32 {vc4value.arg_name = "extent", vc4value.scalar_role = "extent"},
      %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["extent"]})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @shape_duplicate argument #2 'x': vc4value.shape_args must not contain duplicate names}}
  func.func @shape_duplicate(
      %m: index {vc4value.arg_name = "m", vc4value.scalar_role = "extent"},
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %x: memref<?x?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["m", "m"]})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @shape_malformed argument #1 'x': vc4value.shape_args must be an ArrayAttr of StringAttr}}
  func.func @shape_malformed(
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = "n"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @stride_missing_arg argument #2 'x': vc4value.stride_args entry 'missing' must name an existing scalar argument}}
  func.func @stride_missing_arg(
      %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
      %s: index {vc4value.arg_name = "s", vc4value.scalar_role = "stride"},
      %x: memref<?xf32, strided<[?]>, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["n"], vc4value.stride_args = ["missing"]})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} { return }
}

// -----

builtin.module {
  // expected-error @+1 {{public value kernel @dim_missing_mapping argument #0 'x': dynamic public memref argument requires vc4value.shape_args}}
  func.func @dim_missing_mapping(
      %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{memref.dim in Phase 4 value ABI: dynamic dimension requires vc4value.shape_args mapping}}
    %d = memref.dim %x, %c0 : memref<?xf32, #vc4value.global>
    func.return
  }
}

// -----

builtin.module {
  func.func @kernel(%x: memref<16xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %c0 = arith.constant 0 : index
    // expected-error @+1 {{direct memref side-effect operation is not legal in the VC4 value surface; use structured vector transfer/planning forms}}
    %v = memref.load %x[%c0] : memref<16xf32, #vc4value.global>
    func.return
  }
}

// -----

builtin.module {
  func.func @kernel(%x: memref<16xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "out"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.000000e+00 : f32
    // expected-error @+1 {{direct memref side-effect operation is not legal in the VC4 value surface; use structured vector transfer/planning forms}}
    memref.store %zero, %x[%c0] : memref<16xf32, #vc4value.global>
    func.return
  }
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    // expected-error @+1 {{vc4value launch axis must be within the enclosing vc4value.grid_rank}}
    %pid = vc4value.program_id {axis = 1 : i32} : index
    func.return
  }
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32} {
    // expected-error @+1 {{vc4value launch axis must be within the enclosing vc4value.grid_rank}}
    %np = vc4value.num_programs {axis = 2 : i32} : index
    func.return
  }
}

// -----

builtin.module {
  func.func @kernel() attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    // expected-error @+1 {{'vc4value.program_id' op axis must be 0, 1, or 2}}
    %pid = vc4value.program_id {axis = -1 : i32} : index
    func.return
  }
}
