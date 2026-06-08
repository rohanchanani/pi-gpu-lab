// RUN: vc4-opt %s | FileCheck %s

module {
  func.func @global_memrefs(
      %x: memref<?xf32, #vc4value.global>,
      %y: memref<?xi32, #vc4value.global>,
      %z: memref<?x?xf32, #vc4value.global>) {
    return
  }
}

// CHECK: memref<?xf32, #vc4value.global>
// CHECK: memref<?xi32, #vc4value.global>
// CHECK: memref<?x?xf32, #vc4value.global>
