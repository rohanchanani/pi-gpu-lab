// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @approx_sfu_rsqrt_f32_lowers(
    %out: memref<16xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %idx: index {vc4value.arg_name = "idx"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite_positive"} {
  %x = arith.constant dense<4.000000e+00> : vector<16xf32>
  %y = math.rsqrt %x : vector<16xf32>
  vector.transfer_write %y, %out[%idx] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @approx_sfu_rsqrt_f32_lowers
// CHECK: vc4kernel.fragment_sfu
// CHECK-SAME: domain = #vc4kernel.fp_domain<finite_positive>
// CHECK-SAME: fp_policy = #vc4kernel.fp_math_policy<approx_sfu>
// CHECK-SAME: kind = #vc4kernel.sfu_kind<rsqrt>
// CHECK-NOT: vc4kernel.fragment_alu.mul
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: math.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
