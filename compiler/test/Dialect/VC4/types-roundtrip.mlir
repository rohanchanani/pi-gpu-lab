// RUN: vc4-opt %s | FileCheck %s

// CHECK: vc4.module @types {
// CHECK: vc4.func private @opaque_types(!vc4.async.token, !vc4.tmu.desc, !vc4.vpm.desc) -> !vc4.dma.desc attributes {form = 0 : i32, threading = 0 : i32}

vc4.module @types {
  vc4.func private @opaque_types(%token: !vc4.async.token, %tmu: !vc4.tmu.desc, %vpm: !vc4.vpm.desc) -> !vc4.dma.desc attributes {threading = 0 : i32, form = 0 : i32}
}
