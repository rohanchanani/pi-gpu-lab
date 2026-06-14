// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

func.func @softmax_v0_composite_natural_exp_lowers(
    %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree",
                vc4value.max_policy = "finite",
                vc4value.softmax_v0 = "one_block_active_1_to_16"} {
  %c0 = arith.constant 0 : index
  %mask = vector.create_mask %n : vector<16xi1>
  %x = arith.constant dense<1.000000e+00> : vector<16xf32>
  %low = arith.constant dense<-8.000000e+01> : vector<16xf32>
  %zero = arith.constant dense<0.000000e+00> : vector<16xf32>
  %active_x = arith.select %mask, %x, %low : vector<16xi1>, vector<16xf32>
  %max = vector.reduction <maxnumf>, %active_x : vector<16xf32> into f32
  %maxv = vector.broadcast %max : f32 to vector<16xf32>
  %shifted = arith.subf %x, %maxv : vector<16xf32>
  %expv = math.exp %shifted : vector<16xf32>
  %active_e = arith.select %mask, %expv, %zero : vector<16xi1>, vector<16xf32>
  %denom = vector.reduction <add>, %active_e : vector<16xf32> into f32
  %one = arith.constant 1.000000e+00 : f32
  %inv = arith.divf %one, %denom : f32
  %invv = vector.broadcast %inv : f32 to vector<16xf32>
  %outv = arith.mulf %active_e, %invv : vector<16xf32>
  vector.transfer_write %outv, %out[%c0], %mask {in_bounds = [true]} : vector<16xf32>, memref<?xf32, #vc4value.global>
  return
}

// CHECK-LABEL: vc4kernel.kernel @softmax_v0_composite_natural_exp_lowers
// CHECK: vc4kernel.fragment_reduce
// CHECK-SAME: kind = #vc4kernel.reduce<fmax>
// CHECK: vc4kernel.fragment_alu.add
// CHECK-SAME: opcode = #vc4kernel.add_alu_opcode<fsub>
// CHECK: vc4kernel.fragment_const {{.*}}1.442695
// CHECK: vc4kernel.fragment_alu.mul
// CHECK-SAME: opcode = #vc4kernel.mul_alu_opcode<fmul>
// CHECK: vc4kernel.fragment_sfu
// CHECK-SAME: kind = #vc4kernel.sfu_kind<exp>
// CHECK: vc4kernel.fragment_reduce
// CHECK-SAME: kind = #vc4kernel.reduce<add>
// CHECK: vc4kernel.fragment_sfu
// CHECK-SAME: kind = #vc4kernel.sfu_kind<recip>
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: math.
// CHECK-NOT: arith.divf
// CHECK-NOT: vector.
// CHECK-NOT: memref.
