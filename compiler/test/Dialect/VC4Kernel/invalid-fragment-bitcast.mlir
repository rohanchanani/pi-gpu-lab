// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_same_i32(%x : i32) attributes {
    public_name = "bad_same_i32",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: fragment_bitcast requires i32/f32 reinterpretation
    %bad = vc4kernel.fragment_bitcast %v : vector<16xi32> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_same_f32(%x : f32) attributes {
    public_name = "bad_same_f32",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : f32 -> vector<16xf32>
    // CHECK: fragment_bitcast requires i32/f32 reinterpretation
    %bad = vc4kernel.fragment_bitcast %v : vector<16xf32> -> vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_scalar(%x : i32) attributes {
    public_name = "bad_scalar",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    // CHECK: operand #0 must be vector<16xi32> or vector<16xf32>
    %bad = vc4kernel.fragment_bitcast %x : i32 -> vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_width(%x : i32) attributes {
    public_name = "bad_width",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = arith.constant dense<0> : vector<8xi32>
    // CHECK: operand #0 must be vector<16xi32> or vector<16xf32>
    %bad = vc4kernel.fragment_bitcast %v : vector<8xi32> -> vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_element_type(%x : i32) attributes {
    public_name = "bad_element_type",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = arith.constant dense<0> : vector<16xi16>
    // CHECK: operand #0 must be vector<16xi32> or vector<16xf32>
    %bad = vc4kernel.fragment_bitcast %v : vector<16xi16> -> vector<16xf32>
    vc4kernel.return
  }
}
