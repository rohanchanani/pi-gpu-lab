// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @missing_amount(%x : i32) attributes {
    public_name = "missing_amount",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: requires exactly one of static amount attr or dynamic i32 amount operand
    %bad = vc4kernel.fragment_rotate %v : vector<16xi32> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @both_amounts(%x : i32) attributes {
    public_name = "both_amounts",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: requires exactly one of static amount attr or dynamic i32 amount operand
    %bad = vc4kernel.fragment_rotate %v, %x {amount = 1 : i32} : vector<16xi32>, i32 -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_static_amount(%x : i32) attributes {
    public_name = "bad_static_amount",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: amount must be in range [0, 15]
    %bad = vc4kernel.fragment_rotate %v {amount = 16 : i32} : vector<16xi32> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_vector_amount(%x : i32) attributes {
    public_name = "bad_vector_amount",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: operand #1 must be i32
    %bad = vc4kernel.fragment_rotate %v, %v : vector<16xi32>, vector<16xi32> -> vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_result_type(%x : i32) attributes {
    public_name = "bad_result_type",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
  // CHECK: fragment_rotate result must have matching types
    %bad = vc4kernel.fragment_rotate %v {amount = 1 : i32} : vector<16xi32> -> vector<16xf32>
    vc4kernel.return
  }
}
