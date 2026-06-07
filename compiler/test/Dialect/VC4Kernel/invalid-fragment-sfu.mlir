// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @missing_policy(%x : f32) attributes {
    public_name = "missing_policy",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : f32 -> vector<16xf32>
    // CHECK: fragment_sfu requires explicit approximate SFU math policy
    %bad = vc4kernel.fragment_sfu %v {kind = #vc4kernel.sfu_kind<recip>, domain = #vc4kernel.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @exact_policy(%x : f32) attributes {
    public_name = "exact_policy",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : f32 -> vector<16xf32>
    // CHECK: exact floating-point math cannot be lowered to VC4 SFU
    %bad = vc4kernel.fragment_sfu %v {kind = #vc4kernel.sfu_kind<recip>, fp_policy = #vc4kernel.fp_math_policy<exact>, domain = #vc4kernel.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @wrong_domain(%x : f32) attributes {
    public_name = "wrong_domain",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : f32 -> vector<16xf32>
    // CHECK: fragment_sfu domain does not match SFU kind contract
    %bad = vc4kernel.fragment_sfu %v {kind = #vc4kernel.sfu_kind<rsqrt>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_i32(%x : i32) attributes {
    public_name = "bad_i32",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "i32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : i32 -> vector<16xi32>
    // CHECK: operand #0 must be vector<16xf32>
    %bad = vc4kernel.fragment_sfu %v {kind = #vc4kernel.sfu_kind<recip>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_nonzero>} : vector<16xi32> -> vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_width(%x : f32) attributes {
    public_name = "bad_width",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %v = arith.constant dense<0.000000e+00> : vector<8xf32>
    // CHECK: operand #0 must be vector<16xf32>
    %bad = vc4kernel.fragment_sfu %v {kind = #vc4kernel.sfu_kind<recip>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_nonzero>} : vector<8xf32> -> vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_scalar(%x : f32) attributes {
    public_name = "bad_scalar",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    // CHECK: operand #0 must be vector<16xf32>
    %bad = vc4kernel.fragment_sfu %x {kind = #vc4kernel.sfu_kind<recip>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_nonzero>} : f32 -> vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @unsupported_sqrt(%x : f32) attributes {
    public_name = "unsupported_sqrt",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : f32 -> vector<16xf32>
    // CHECK: expected ::mlir::vc4kernel::SFUKind to be one of: recip, rsqrt, exp, log
    %bad = vc4kernel.fragment_sfu %v {kind = #vc4kernel.sfu_kind<sqrt>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_positive>} : vector<16xf32> -> vector<16xf32>
    vc4kernel.return
  }
}
