// RUN: vc4-opt %s | FileCheck %s

module {
  %zero = ssavc4.load_imm <splat32> {value = 0.000000e+00 : f32} : vector<16xf32>
  // CHECK: ssavc4.sfu
  // CHECK-SAME: domain = #ssavc4.fp_domain<finite_nonzero>
  // CHECK-SAME: fp_policy = #ssavc4.fp_math_policy<approx_sfu>
  // CHECK-SAME: kind = #ssavc4.sfu_kind<recip>
  %recip = ssavc4.sfu %zero {kind = #ssavc4.sfu_kind<recip>, fp_policy = #ssavc4.fp_math_policy<approx_sfu>, domain = #ssavc4.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
  // CHECK: ssavc4.sfu
  // CHECK-SAME: kind = #ssavc4.sfu_kind<rsqrt>
  %rsqrt = ssavc4.sfu %recip {kind = #ssavc4.sfu_kind<rsqrt>, fp_policy = #ssavc4.fp_math_policy<approx_sfu>, domain = #ssavc4.fp_domain<finite_positive>} : vector<16xf32> -> vector<16xf32>
  // CHECK: ssavc4.sfu
  // CHECK-SAME: kind = #ssavc4.sfu_kind<exp>
  %exp = ssavc4.sfu %rsqrt {kind = #ssavc4.sfu_kind<exp>, fp_policy = #ssavc4.fp_math_policy<approx_sfu>, domain = #ssavc4.fp_domain<finite>} : vector<16xf32> -> vector<16xf32>
  // CHECK: ssavc4.sfu
  // CHECK-SAME: kind = #ssavc4.sfu_kind<log>
  %log = ssavc4.sfu %exp {kind = #ssavc4.sfu_kind<log>, fp_policy = #ssavc4.fp_math_policy<approx_sfu>, domain = #ssavc4.fp_domain<finite_positive>} : vector<16xf32> -> vector<16xf32>
}
