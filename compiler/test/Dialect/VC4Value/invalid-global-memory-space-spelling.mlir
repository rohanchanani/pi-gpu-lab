// RUN: not vc4-opt %s 2>&1 | FileCheck %s

module {
  func.func @bad(%x: memref<?xf32, #vc4value.glob>) {
    return
  }
}

// CHECK: error:
