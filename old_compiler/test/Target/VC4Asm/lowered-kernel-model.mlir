// RUN: vc4-opt --lower-gpu-to-vc4 %s | vc4-translate --mlir-to-vc4-kernel-model | FileCheck %s

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

// CHECK: kernel @saxpy
// CHECK: public_args:
// CHECK:   arg0: memref<?xf32>
// CHECK:   arg1: memref<?xf32>
// CHECK:   arg2: f32
// CHECK:   arg3: index
// CHECK: uniform_stream:
// CHECK:   uniform0: arg0 : memref<?xf32>
// CHECK:   uniform1: arg1 : memref<?xf32>
// CHECK:   uniform2: arg2 : vector<16xf32>
// CHECK:   uniform3: arg3 : index
// CHECK:   uniform4: builtin qpu_id : index
// CHECK:   uniform5: builtin num_qpus : index
// CHECK: builtin_suffix_start: 4
// CHECK: execution:
// CHECK:   worker_id_builtin: qpu_id
// CHECK:   worker_count_builtin: num_qpus
// CHECK:   lane_width: 16
// CHECK:   base = qpu_id * 16
// CHECK:   stride = num_qpus * 16
// CHECK:   upper_bound: uniform3
// CHECK:   assumes_upper_bound_multiple_of_lane_width: true
// CHECK: body:
// CHECK:   dma_load uniform0[iv] -> stage0
// CHECK:   staged_read stage0 -> t0
// CHECK:   dma_load uniform1[iv] -> stage1
// CHECK:   staged_read stage1 -> t1
// CHECK:   fmul uniform2, t0 -> t2
// CHECK:   fadd t2, t1 -> t3
// CHECK:   staged_write t3 -> stage2
// CHECK:   dma_store stage2 -> uniform1[iv]
// CHECK-NOT: row
