// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @policy_missing(%x : f32) attributes {
    public_name = "policy_missing",
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
  vc4kernel.kernel @policy_exact(%x : f32) attributes {
    public_name = "policy_exact",
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
  vc4kernel.kernel @policy_wrong_domain(%x : f32) attributes {
    public_name = "policy_wrong_domain",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : f32 -> vector<16xf32>
    // CHECK: fragment_sfu domain does not match SFU kind contract
    %bad = vc4kernel.fragment_sfu %v {kind = #vc4kernel.sfu_kind<log>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite>} : vector<16xf32> -> vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @policy_sqrt_reject(%x : f32) attributes {
    public_name = "policy_sqrt_reject",
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
