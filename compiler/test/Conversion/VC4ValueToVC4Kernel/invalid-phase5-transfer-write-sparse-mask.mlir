// RUN: not vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @write_sparse_mask(%out: memref<64xi32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
                             %base: index {vc4value.arg_name = "base"},
                             %limit: index {vc4value.arg_name = "limit"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %lanes = vector.step : vector<16xindex>
  %limit_v = vector.broadcast %limit : index to vector<16xindex>
  %mask = arith.cmpi ult, %lanes, %limit_v : vector<16xindex>
  %value = arith.constant dense<1> : vector<16xi32>
  vector.transfer_write %value, %out[%base], %mask {in_bounds = [true]} : vector<16xi32>, memref<64xi32, #vc4value.global>
  return
}

// CHECK: sparse or unknown transfer_write mask is not Phase 5 lowerable
// CHECK: staged value-surface feature
// CHECK: READY_FOR_TRITON remains NO
