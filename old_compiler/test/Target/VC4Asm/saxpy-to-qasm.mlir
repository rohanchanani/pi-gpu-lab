// RUN: vc4-opt --lower-gpu-to-vc4 %s | vc4-translate --mlir-to-vc4-qasm | FileCheck %s

gpu.module @kernels {
  gpu.func @saxpy(%x: memref<?xf32>, %y: memref<?xf32>, %a: f32, %n: index) kernel {
    %gid = gpu.global_id x
    %inBounds = arith.cmpi ult, %gid, %n : index
    scf.if %inBounds {
      %xval = memref.load %x[%gid] : memref<?xf32>
      %yval = memref.load %y[%gid] : memref<?xf32>
      %mul = arith.mulf %a, %xval : f32
      %sum = arith.addf %mul, %yval : f32
      memref.store %sum, %y[%gid] : memref<?xf32>
    }
    gpu.return
  }
}

// CHECK: .include "share/vc4inc/vc4.qinc"
// CHECK: # kernel @saxpy
// CHECK: mov ra0, unif
// CHECK: mov ra1, unif
// CHECK: mov ra2, unif
// CHECK: mov ra3, unif
// CHECK: mov ra4, unif
// CHECK: mov ra5, unif
// CHECK: shl r0, ra4, 6
// CHECK: mov ra6, r0
// CHECK: shl r0, ra5, 6
// CHECK: mov ra7, r0
// CHECK: shl r0, ra3, 2
// CHECK: mov ra8, r0
// CHECK: brr.anync -, :end
// CHECK: :loop
// CHECK: mov r2, vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0))
// CHECK: mov vr_setup, r2
// CHECK: add vr_addr, ra0, ra6
// CHECK: mov -, vr_wait
// CHECK: mov r2, vpm_setup(1, 1, h32(0))
// CHECK: mov vr_setup, r2
// CHECK: mov rb0, vpm
// CHECK: mov -, vw_wait
// CHECK: mov r2, vdr_setup_0(0, 16, 1, vdr_h32(1, 0, 0))
// CHECK: add vr_setup, r2, 16
// CHECK: add vr_addr, ra1, ra6
// CHECK: mov -, vr_wait
// CHECK: mov r2, vpm_setup(1, 1, h32(0))
// CHECK: add vr_setup, r2, 1
// CHECK: mov rb1, vpm
// CHECK: mov -, vw_wait
// CHECK: fmul rb2, ra2, rb0
// CHECK: fadd rb3, rb2, rb1
// CHECK: mov r2, vpm_setup(1, 1, h32(0))
// CHECK: add vw_setup, r2, 2
// CHECK: mov vpm, rb3
// CHECK: mov -, vw_wait
// CHECK: mov r2, vdw_setup_0(1, 16, dma_h32(0, 0))
// CHECK: add vw_setup, r2, 256
// CHECK: add vw_addr, ra1, ra6
// CHECK: mov -, vw_wait
// CHECK: add ra6, ra6, ra7
// CHECK: brr.anyc -, :loop
// CHECK: :end
// CHECK: thrend
// CHECK: mov interrupt, 1
