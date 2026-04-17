// RUN: vc4-opt --lower-gpu-to-vc4 %s | vc4-translate --mlir-to-vc4-kernel-model | FileCheck %s

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

// CHECK: kernel @blend_linear
// CHECK: public_args:
// CHECK:   arg0: memref<?xf32>
// CHECK:   arg1: memref<?xf32>
// CHECK:   arg2: memref<?xf32>
// CHECK:   arg3: f32
// CHECK:   arg4: f32
// CHECK:   arg5: index
// CHECK: uniform_stream:
// CHECK:   uniform0: arg0 : memref<?xf32>
// CHECK:   uniform1: arg1 : memref<?xf32>
// CHECK:   uniform2: arg2 : memref<?xf32>
// CHECK:   uniform3: arg3 : vector<16xf32>
// CHECK:   uniform4: arg4 : vector<16xf32>
// CHECK:   uniform5: arg5 : index
// CHECK:   uniform6: builtin qpu_id : index
// CHECK:   uniform7: builtin num_qpus : index
// CHECK: builtin_suffix_start: 6
// CHECK: execution:
// CHECK:   worker_id_builtin: qpu_id
// CHECK:   worker_count_builtin: num_qpus
// CHECK:   lane_width: 16
// CHECK:   base = qpu_id * 16
// CHECK:   stride = num_qpus * 16
// CHECK:   upper_bound: uniform5
// CHECK: body:
// CHECK:   dma_load uniform0[iv] -> stage0
// CHECK:   staged_read stage0 -> t0
// CHECK:   dma_load uniform1[iv] -> stage1
// CHECK:   staged_read stage1 -> t1
// CHECK:   fmul uniform3, t0 -> t2
// CHECK:   fmul uniform4, t1 -> t3
// CHECK:   fadd t2, t3 -> t4
// CHECK:   staged_write t4 -> stage2
// CHECK:   dma_store stage2 -> uniform2[iv]
