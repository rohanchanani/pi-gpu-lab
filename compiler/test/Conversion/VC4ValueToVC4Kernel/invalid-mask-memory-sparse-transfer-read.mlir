// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_sparse_transfer_read(
    %in: memref<64xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %base: index {vc4value.arg_name = "base"},
    %limit: index {vc4value.arg_name = "limit"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %zero = arith.constant 0 : i32
  %lanes = vector.step : vector<16xindex>
  %limit_v = vector.broadcast %limit : index to vector<16xindex>
  %mask = arith.cmpi ult, %lanes, %limit_v : vector<16xindex>
  %v = vector.transfer_read %in[%base], %zero, %mask {in_bounds = [true]} : memref<64xi32, #vc4value.global>, vector<16xi32>
  return
}

// CHECK: sparse or unknown transfer_read mask is not Phase 5 lowerable
// CHECK: staged value-surface feature
// CHECK: READY_FOR_TRITON remains NO
