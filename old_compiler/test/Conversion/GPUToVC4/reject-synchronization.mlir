// RUN: not vc4-opt --lower-gpu-to-vc4 %s 2>&1 | FileCheck %s

gpu.module @kernels {
  gpu.func @barrier_kernel(%src: memref<?xf32>, %dst: memref<?xf32>,
                           %scale: f32, %n: index) kernel {
    %gid = gpu.global_id x
    %inBounds = arith.cmpi ult, %gid, %n : index
    scf.if %inBounds {
      %lhs = memref.load %src[%gid] : memref<?xf32>
      gpu.barrier
      %mul = arith.mulf %scale, %lhs : f32
      memref.store %mul, %dst[%gid] : memref<?xf32>
    }
    gpu.return
  }
}

// CHECK: error:
// CHECK-SAME: barriers and inter-thread synchronization are outside the currently supported gpu-to-vc4 subset
