// RUN: not vc4-opt --lower-gpu-to-vc4 %s 2>&1 | FileCheck %s

gpu.module @kernels {
  gpu.func @unguarded_linear(%src: memref<?xf32>, %dst: memref<?xf32>,
                             %scale: f32, %extent: index) kernel {
    %gid = gpu.global_id x
    %lhs = memref.load %src[%gid] : memref<?xf32>
    %scaled = arith.mulf %scale, %lhs : f32
    memref.store %scaled, %dst[%gid] : memref<?xf32>
    gpu.return
  }
}

// CHECK: error:
// CHECK-SAME: unguarded kernels are not yet supported because the logical upper bound is not explicit to the backend
