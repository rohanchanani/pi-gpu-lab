// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel -o %t/value_softmax.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VC4KERNEL --input-file=%t/value_softmax.vc4kernel.mlir
// RUN: vc4-opt %t/value_softmax.vc4kernel.mlir --verify-vc4kernel --convert-vc4kernel-to-ssavc4 -o %t/value_softmax.ssavc4.mlir
// RUN: vc4-opt %t/value_softmax.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/value_softmax.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/value_softmax.vc4.mlir

func.func @value_softmax_stable_f32_b16_vc4value(
    %x: memref<?xf32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["n"]},
    %y: memref<?xf32, #vc4value.global> {vc4value.arg_name = "y", vc4value.direction = "out", vc4value.shape_args = ["n"]},
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree",
                vc4value.max_policy = "finite",
                vc4value.softmax_v0 = "one_block_active_1_to_16"} {
  %c0 = arith.constant 0 : index
  %mask = vector.create_mask %n : vector<16xi1>
  %zero = arith.constant 0.000000e+00 : f32
  %xv = vector.transfer_read %x[%c0], %zero, %mask {in_bounds = [true]} : memref<?xf32, #vc4value.global>, vector<16xf32>
  %low = arith.constant dense<-8.000000e+01> : vector<16xf32>
  %zero_v = arith.constant dense<0.000000e+00> : vector<16xf32>
  %active_x = arith.select %mask, %xv, %low : vector<16xi1>, vector<16xf32>
  %max = vector.reduction <maxnumf>, %active_x : vector<16xf32> into f32
  %maxv = vector.broadcast %max : f32 to vector<16xf32>
  %shifted = arith.subf %xv, %maxv : vector<16xf32>
  %expv = math.exp %shifted : vector<16xf32>
  %active_e = arith.select %mask, %expv, %zero_v : vector<16xi1>, vector<16xf32>
  %denom = vector.reduction <add>, %active_e : vector<16xf32> into f32
  %one = arith.constant 1.000000e+00 : f32
  %inv = arith.divf %one, %denom : f32
  %invv = vector.broadcast %inv : f32 to vector<16xf32>
  %outv = arith.mulf %active_e, %invv : vector<16xf32>
  vector.transfer_write %outv, %y[%c0], %mask {in_bounds = [true]} : vector<16xf32>, memref<?xf32, #vc4value.global>
  return
}

// VC4KERNEL-LABEL: vc4kernel.kernel @value_softmax_stable_f32_b16_vc4value
// VC4KERNEL: vc4kernel.fragment_reduce
// VC4KERNEL-SAME: kind = #vc4kernel.reduce<fmax>
// VC4KERNEL: vc4kernel.fragment_sfu
// VC4KERNEL-SAME: kind = #vc4kernel.sfu_kind<exp>
// VC4KERNEL: vc4kernel.fragment_sfu
// VC4KERNEL-SAME: kind = #vc4kernel.sfu_kind<recip>
// VC4: vc4.module
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.
