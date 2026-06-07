// RUN: vc4-opt %s -split-input-file -verify-diagnostics

module {
  %x = ssavc4.load_imm <splat32> {value = 1.000000e+00 : f32} : vector<16xf32>
  // expected-error@+1 {{exact floating-point math cannot be lowered to VC4 SFU}}
  %bad = ssavc4.sfu %x {kind = #ssavc4.sfu_kind<recip>, fp_policy = #ssavc4.fp_math_policy<exact>, domain = #ssavc4.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
}

// -----

module {
  %x = ssavc4.load_imm <splat32> {value = 1.000000e+00 : f32} : vector<16xf32>
  // expected-error@+1 {{sfu domain does not match SFU kind contract}}
  %bad = ssavc4.sfu %x {kind = #ssavc4.sfu_kind<log>, fp_policy = #ssavc4.fp_math_policy<approx_sfu>, domain = #ssavc4.fp_domain<finite>} : vector<16xf32> -> vector<16xf32>
}

// -----

module {
  %x = ssavc4.load_imm <splat32> {value = 1 : i32} : vector<16xi32>
  // expected-error@+1 {{input type must be an SSAVC4 float carrier}}
  %bad = ssavc4.sfu %x {kind = #ssavc4.sfu_kind<recip>, fp_policy = #ssavc4.fp_math_policy<approx_sfu>, domain = #ssavc4.fp_domain<finite_nonzero>} : vector<16xi32> -> vector<16xf32>
}

// -----

module {
  %x = arith.constant dense<0.000000e+00> : vector<8xf32>
  // expected-error@+1 {{input type must be vector<16xi32> or vector<16xf32>}}
  %bad = ssavc4.sfu %x {kind = #ssavc4.sfu_kind<recip>, fp_policy = #ssavc4.fp_math_policy<approx_sfu>, domain = #ssavc4.fp_domain<finite_nonzero>} : vector<8xf32> -> vector<16xf32>
}

// -----

module {
  %x = ssavc4.load_imm <splat32> {value = 1.000000e+00 : f32} : vector<16xf32>
  // expected-error@+2 {{expected ::mlir::ssavc4::SFUKind to be one of: recip, rsqrt, exp, log}}
  // expected-error@+1 {{failed to parse SSAVC4_SFUKindAttr parameter 'value'}}
  %bad = ssavc4.sfu %x {kind = #ssavc4.sfu_kind<sqrt>, fp_policy = #ssavc4.fp_math_policy<approx_sfu>, domain = #ssavc4.fp_domain<finite_positive>} : vector<16xf32> -> vector<16xf32>
}
