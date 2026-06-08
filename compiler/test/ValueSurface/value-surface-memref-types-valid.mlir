// RUN: vc4-opt %s --vc4-verify-value-surface | FileCheck %s

builtin.module {
  // CHECK-LABEL: func.func @kernel
  func.func @kernel(
      %m_i8: memref<16xi8> {vc4value.arg_name = "m_i8", vc4value.direction = "in"},
      %m_i16: memref<4x4xi16> {vc4value.arg_name = "m_i16", vc4value.direction = "in"},
      %m_i32: memref<16xi32> {vc4value.arg_name = "m_i32", vc4value.direction = "in"},
      %m_f16: memref<4x4xf16> {vc4value.arg_name = "m_f16", vc4value.direction = "in"},
      %m_f32: memref<16xf32> {vc4value.arg_name = "m_f32", vc4value.direction = "in"})
      attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
    %pid = vc4value.program_id {axis = 0 : i32} : index
    %c0 = arith.constant 0 : index
    // CHECK: memref.dim
    %dim = memref.dim %m_f32, %c0 : memref<16xf32>
    // CHECK: memref.cast
    %cast = memref.cast %m_f32 : memref<16xf32> to memref<?xf32>
    return
  }
}
