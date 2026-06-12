// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

func.func @invalid_rank2_transfer(
    %in: memref<4x16xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %i: index {vc4value.arg_name = "i"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %zero = arith.constant 0 : i32
  %v = vector.transfer_read %in[%i, %i], %zero {in_bounds = [true]} : memref<4x16xi32, #vc4value.global>, vector<16xi32>
  return
}

// CHECK: ranked or strided memory beyond Phase 10
// CHECK: expected rank-1 contiguous i32/f32/f16 #vc4value.global memref

// -----

func.func @invalid_strided_transfer(
    %in: memref<?xf32, strided<[?]>, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.000000e+00 : f32
  %v = vector.transfer_read %in[%c0], %zero {in_bounds = [true]} : memref<?xf32, strided<[?]>, #vc4value.global>, vector<16xf32>
  return
}

// CHECK: READY_FOR_TRITON remains NO
