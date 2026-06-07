// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.func @fragment_sfu_lowering
// CHECK: ssavc4.sfu
// CHECK-SAME: domain = #ssavc4.fp_domain<finite_nonzero>
// CHECK-SAME: fp_policy = #ssavc4.fp_math_policy<approx_sfu>
// CHECK-SAME: kind = #ssavc4.sfu_kind<recip>
// CHECK: ssavc4.sfu
// CHECK-SAME: domain = #ssavc4.fp_domain<finite_positive>
// CHECK-SAME: kind = #ssavc4.sfu_kind<rsqrt>
// CHECK: ssavc4.sfu
// CHECK-SAME: domain = #ssavc4.fp_domain<finite>
// CHECK-SAME: kind = #ssavc4.sfu_kind<exp>
// CHECK: ssavc4.sfu
// CHECK-SAME: domain = #ssavc4.fp_domain<finite_positive>
// CHECK-SAME: kind = #ssavc4.sfu_kind<log>
// CHECK-NOT: vc4.qpu.bundle
// CHECK-NOT: vc4kernel.
module {
  vc4kernel.kernel @fragment_sfu_lowering(%x : f32) attributes {
    public_name = "fragment_sfu_lowering",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [{name = "x", kind = "scalar", direction = "by_value", type = "f32"}],
    warps_per_block = 1 : i32
  } {
    %v = vc4kernel.splat %x : f32 -> vector<16xf32>
    %recip = vc4kernel.fragment_sfu %v {kind = #vc4kernel.sfu_kind<recip>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_nonzero>} : vector<16xf32> -> vector<16xf32>
    %rsqrt = vc4kernel.fragment_sfu %recip {kind = #vc4kernel.sfu_kind<rsqrt>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_positive>} : vector<16xf32> -> vector<16xf32>
    %exp = vc4kernel.fragment_sfu %rsqrt {kind = #vc4kernel.sfu_kind<exp>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite>} : vector<16xf32> -> vector<16xf32>
    %log = vc4kernel.fragment_sfu %exp {kind = #vc4kernel.sfu_kind<log>, fp_policy = #vc4kernel.fp_math_policy<approx_sfu>, domain = #vc4kernel.fp_domain<finite_positive>} : vector<16xf32> -> vector<16xf32>
    vc4kernel.return
  }
}
