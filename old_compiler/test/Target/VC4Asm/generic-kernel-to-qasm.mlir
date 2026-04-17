// RUN: vc4-opt --lower-gpu-to-vc4 %s | vc4-translate --mlir-to-vc4-qasm | FileCheck %s

gpu.module @kernels {
  gpu.func @blend_linear(%lhs: memref<?xf32>, %rhs: memref<?xf32>,
                         %out: memref<?xf32>, %alpha: f32,
                         %beta: f32, %extent: index) kernel {
    %lane = gpu.global_id x
    %inBounds = arith.cmpi ult, %lane, %extent : index
    scf.if %inBounds {
      %lhsv = memref.load %lhs[%lane] : memref<?xf32>
      %rhsv = memref.load %rhs[%lane] : memref<?xf32>
      %lhsScaled = arith.mulf %alpha, %lhsv : f32
      %rhsScaled = arith.mulf %beta, %rhsv : f32
      %sum = arith.addf %lhsScaled, %rhsScaled : f32
      memref.store %sum, %out[%lane] : memref<?xf32>
    }
    gpu.return
  }
}

// CHECK: .include "share/vc4inc/vc4.qinc"
// CHECK: # kernel @blend_linear
// CHECK: mov ra0, unif
// CHECK: mov ra1, unif
// CHECK: mov ra2, unif
// CHECK: mov ra3, unif
// CHECK: mov ra4, unif
// CHECK: mov ra5, unif
// CHECK: mov ra6, unif
// CHECK: mov ra7, unif
// CHECK: shl r0, ra6, 6
// CHECK: mov ra8, r0
// CHECK: shl r0, ra7, 6
// CHECK: mov ra9, r0
// CHECK: shl r0, ra5, 2
// CHECK: mov ra10, r0
// CHECK: brr.anync -, :end
// CHECK: :loop
// CHECK: add vr_addr, ra0, ra8
// CHECK: mov rb0, vpm
// CHECK: add vr_addr, ra1, ra8
// CHECK: mov rb1, vpm
// CHECK: fmul rb2, ra3, rb0
// CHECK: fmul rb3, ra4, rb1
// CHECK: fadd rb4, rb2, rb3
// CHECK: mov vpm, rb4
// CHECK: add vw_addr, ra2, ra8
// CHECK: add ra8, ra8, ra9
// CHECK: brr.anyc -, :loop
// CHECK: thrend
