// RUN: not vc4-opt %s --convert-vc4-value-to-vc4kernel --verify-vc4kernel 2>&1 | FileCheck %s

func.func @invalid_hidden_memref_descriptor_dim(
    %x: memref<?x?xi32, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"]},
    %rows: index {vc4value.arg_name = "rows"},
    %cols: index {vc4value.arg_name = "cols"},
    %which: index {vc4value.arg_name = "which"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %d = memref.dim %x, %which : memref<?x?xi32, #vc4value.global>
  return
}

// CHECK: memref.dim hidden descriptor ABI is rejected
// CHECK: dimension index must be constant
// CHECK: READY_FOR_TRITON remains NO
