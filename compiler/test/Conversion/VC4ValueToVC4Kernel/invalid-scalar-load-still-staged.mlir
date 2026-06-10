// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_scalar_load_still_staged(
    %in: memref<16xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %idx: index {vc4value.arg_name = "idx"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %value = memref.load %in[%idx] : memref<16xi32, #vc4value.global>
  return
}

// CHECK: scalar memref.load is staged
