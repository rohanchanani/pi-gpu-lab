// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_scalar_store_unsupported_type(
    %out: memref<16xf16, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %value: f16 {vc4value.arg_name = "value"},
    %idx: index {vc4value.arg_name = "idx"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  memref.store %value, %out[%idx] : memref<16xf16, #vc4value.global>
  return
}

// CHECK: unsupported transfer element type
