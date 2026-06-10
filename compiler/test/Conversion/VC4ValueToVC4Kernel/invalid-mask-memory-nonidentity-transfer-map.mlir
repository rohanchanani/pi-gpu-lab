// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_nonidentity_read_map(
    %in: memref<64xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %base: index {vc4value.arg_name = "base"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %zero = arith.constant 0 : i32
  %v = vector.transfer_read %in[%base], %zero {permutation_map = affine_map<(d0) -> (0)>} : memref<64xi32, #vc4value.global>, vector<16xi32>
  return
}

// CHECK: transfer_read permutation map beyond rank-1 identity is not Phase 5 lowerable
// CHECK: READY_FOR_TRITON remains NO
