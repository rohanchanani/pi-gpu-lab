// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @approx_sfu_recip_div_f32_lowers(
    %outv: memref<16xf32, #vc4value.global> {vc4value.arg_name = "outv", vc4value.direction = "out"},
    %outs: memref<16xf32, #vc4value.global> {vc4value.arg_name = "outs", vc4value.direction = "out"},
    %idx: index {vc4value.arg_name = "idx"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite_nonzero"} {
  %num = arith.constant dense<4.000000e+00> : vector<16xf32>
  %den = arith.constant dense<2.000000e+00> : vector<16xf32>
  %y = arith.divf %num, %den : vector<16xf32>
  %one = arith.constant 1.000000e+00 : f32
  %two = arith.constant 2.000000e+00 : f32
  %inv = arith.divf %one, %two : f32
  vector.transfer_write %y, %outv[%idx] {in_bounds = [true]} : vector<16xf32>, memref<16xf32, #vc4value.global>
  memref.store %inv, %outs[%idx] : memref<16xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @approx_sfu_recip_div_f32_lowers
// CHECK: vc4kernel.fragment_sfu
// CHECK-SAME: domain = #vc4kernel.fp_domain<finite_nonzero>
// CHECK-SAME: fp_policy = #vc4kernel.fp_math_policy<approx_sfu>
// CHECK-SAME: kind = #vc4kernel.sfu_kind<recip>
// CHECK: vc4kernel.fragment_alu.mul
// CHECK-SAME: opcode = #vc4kernel.mul_alu_opcode<fmul>
// CHECK: vc4kernel.fragment_sfu
// CHECK-SAME: kind = #vc4kernel.sfu_kind<recip>
// CHECK-NOT: arith.divf
// CHECK-NOT: math.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
