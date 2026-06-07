// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s

module {
  vc4kernel.kernel @fragment_sfu(%x : f32) attributes {
    public_name = "fragment_sfu",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : f32 -> vector<16xf32>
    // CHECK: vc4kernel.fragment_sfu
    // CHECK-SAME: domain = #vc4kernel.fp_domain<finite_nonzero>
    // CHECK-SAME: fp_policy = #vc4kernel.fp_math_policy<approx_sfu>
    // CHECK-SAME: kind = #vc4kernel.sfu_kind<recip>
    %recip = vc4kernel.fragment_sfu %v {kind = #vc4kernel.sfu_kind<recip>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
    // CHECK: vc4kernel.fragment_sfu
    // CHECK-SAME: domain = #vc4kernel.fp_domain<finite_positive>
    // CHECK-SAME: kind = #vc4kernel.sfu_kind<rsqrt>
    %rsqrt = vc4kernel.fragment_sfu %recip {kind = #vc4kernel.sfu_kind<rsqrt>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_positive>} : vector<16xf32> -> vector<16xf32>
    // CHECK: vc4kernel.fragment_sfu
    // CHECK-SAME: domain = #vc4kernel.fp_domain<finite>
    // CHECK-SAME: kind = #vc4kernel.sfu_kind<exp>
    %exp = vc4kernel.fragment_sfu %rsqrt {kind = #vc4kernel.sfu_kind<exp>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite>} : vector<16xf32> -> vector<16xf32>
    // CHECK: vc4kernel.fragment_sfu
    // CHECK-SAME: domain = #vc4kernel.fp_domain<finite_positive>
    // CHECK-SAME: kind = #vc4kernel.sfu_kind<log>
    %log = vc4kernel.fragment_sfu %exp {kind = #vc4kernel.sfu_kind<log>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_positive>} : vector<16xf32> -> vector<16xf32>
    vc4kernel.return
  }
}
