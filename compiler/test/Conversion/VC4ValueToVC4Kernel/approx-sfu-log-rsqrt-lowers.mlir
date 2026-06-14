// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @approx_sfu_log_rsqrt_lowers(
    %out: memref<16xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %idx: index {vc4value.arg_name = "idx"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite_positive"} {
  %x = arith.constant dense<4.000000e+00> : vector<16xf32>
  %l = math.log %x : vector<16xf32>
  %r = math.rsqrt %x : vector<16xf32>
  %sum = arith.addf %l, %r : vector<16xf32>
  vector.transfer_write %sum, %out[%idx] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @approx_sfu_log_rsqrt_lowers
// CHECK: vc4kernel.fragment_sfu
// CHECK-SAME: domain = #vc4kernel.fp_domain<finite_positive>
// CHECK-SAME: kind = #vc4kernel.sfu_kind<log>
// CHECK: vc4kernel.fragment_sfu
// CHECK-SAME: kind = #vc4kernel.sfu_kind<rsqrt>
// CHECK-NOT: math.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
