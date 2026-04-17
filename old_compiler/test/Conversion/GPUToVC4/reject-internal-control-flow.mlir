// RUN: not vc4-opt --lower-gpu-to-vc4 %s 2>&1 | FileCheck %s

gpu.module @kernels {
  gpu.func @nested_if(%src: memref<?xf32>, %dst: memref<?xf32>,
                      %scale: f32, %n: index) kernel {
    %gid = gpu.global_id x
    %inBounds = arith.cmpi ult, %gid, %n : index
    scf.if %inBounds {
      %lhs = memref.load %src[%gid] : memref<?xf32>
      %rhs = memref.load %dst[%gid] : memref<?xf32>
      %pickLhs = arith.cmpi ult, %gid, %n : index
      %sum = scf.if %pickLhs -> (f32) {
        %mul = arith.mulf %scale, %lhs : f32
        scf.yield %mul : f32
      } else {
        scf.yield %rhs : f32
      }
      memref.store %sum, %dst[%gid] : memref<?xf32>
    }
    gpu.return
  }
}

// CHECK: error:
// CHECK-SAME: control-flow predicates and internal control flow are outside the currently supported gpu-to-vc4 straight-line subset
