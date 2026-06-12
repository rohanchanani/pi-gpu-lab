// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

func.func @read_nonidentity_layout(
    %n: index {vc4value.arg_name = "n", vc4value.scalar_role = "extent"},
    %s: i32 {vc4value.arg_name = "s", vc4value.scalar_role = "stride"},
    %in: memref<?xf32, strided<[?]>, #vc4value.global> {vc4value.arg_name = "in",
                                                         vc4value.direction = "in",
                                                         vc4value.shape_args = ["n"],
                                                         vc4value.stride_args = ["s"]})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.000000e+00 : f32
  %v = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<?xf32, strided<[?]>, #vc4value.global>, vector<16xf32>
  return
}

// CHECK: expected rank-1 contiguous i32/f32/f16 #vc4value.global memref
// CHECK: not Phase 5 lowerable
// CHECK: staged value-surface feature

// -----

func.func @read_nonidentity_permutation(%in: memref<64xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.000000e+00 : f32
  %v = vector.transfer_read %in[%c0], %zero {permutation_map = affine_map<(d0) -> (0)>} : memref<64xf32, #vc4value.global>, vector<16xf32>
  return
}

// CHECK: transfer_read permutation map beyond rank-1 identity is not Phase 5 lowerable
// CHECK: READY_FOR_TRITON remains NO
