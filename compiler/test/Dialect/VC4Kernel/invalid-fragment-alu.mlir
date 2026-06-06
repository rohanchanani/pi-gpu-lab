// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_add_nop(%x : i32) attributes {
    public_name = "bad_add_nop",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: nop has no fragment value result in VC4Kernel P1
    %bad = vc4kernel.fragment_alu.add %v, %v {opcode = #vc4kernel.add_alu_opcode<nop>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_mul_nop(%x : i32) attributes {
    public_name = "bad_mul_nop",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: nop has no fragment value result in VC4Kernel P1
    %bad = vc4kernel.fragment_alu.mul %v, %v {opcode = #vc4kernel.mul_alu_opcode<nop>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_unary_arity(%x : i32) attributes {
    public_name = "bad_unary_arity",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: selected ADD-pipe opcode requires exactly one operand
    %bad = vc4kernel.fragment_alu.add %v, %v {opcode = #vc4kernel.add_alu_opcode<clz>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_binary_arity(%x : i32) attributes {
    public_name = "bad_binary_arity",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: selected ADD-pipe opcode requires exactly two operands
    %bad = vc4kernel.fragment_alu.add %v {opcode = #vc4kernel.add_alu_opcode<add>} : (vector<16xi32>) -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_f32_opcode_on_i32(%x : i32) attributes {
    public_name = "bad_f32_opcode_on_i32",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: result type must be vector<16xf32>
    %bad = vc4kernel.fragment_alu.add %v, %v {opcode = #vc4kernel.add_alu_opcode<fadd>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_i32_opcode_on_f32(%x : f32) attributes {
    public_name = "bad_i32_opcode_on_f32",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : f32 -> vector<16xf32>
    // CHECK: result type must be vector<16xi32>
    %bad = vc4kernel.fragment_alu.add %v, %v {opcode = #vc4kernel.add_alu_opcode<and>} : (vector<16xf32>, vector<16xf32>) -> vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_mul_arity(%x : i32) attributes {
    public_name = "bad_mul_arity",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: selected MUL-pipe opcode requires exactly two operands
    %bad = vc4kernel.fragment_alu.mul %v {opcode = #vc4kernel.mul_alu_opcode<mul24>} : (vector<16xi32>) -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_fmul_i32(%x : i32) attributes {
    public_name = "bad_fmul_i32",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: result type must be vector<16xf32>
    %bad = vc4kernel.fragment_alu.mul %v, %v {opcode = #vc4kernel.mul_alu_opcode<fmul>} : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
    vc4kernel.return
  }
}
