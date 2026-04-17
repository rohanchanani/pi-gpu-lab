// RUN: vc4-opt --lower-gpu-to-vc4 %s | FileCheck %s

// Positive conversion coverage for the current feature-bounded subset.
// These kernels vary names, argument roles, and body shapes while staying within
// the supported contract:
// - gpu.func kernel
// - 1D indexing
// - straight-line body
// - canonical in-bounds guard
// - memref.load/memref.store + arith.mulf/arith.addf only

gpu.module @kernels {
  gpu.func @axpy_guarded(%src: memref<?xf32>, %dst: memref<?xf32>,
                         %scale: f32, %limit: index) kernel {
    %tid = gpu.global_id x
    %ok = arith.cmpi ult, %tid, %limit : index
    scf.if %ok {
      %in = memref.load %src[%tid] : memref<?xf32>
      %acc = memref.load %dst[%tid] : memref<?xf32>
      %scaled = arith.mulf %scale, %in : f32
      %out = arith.addf %scaled, %acc : f32
      memref.store %out, %dst[%tid] : memref<?xf32>
    }
    gpu.return
  }

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

// CHECK: gpu.func @axpy_guarded
// CHECK: %[[QPU0:.*]] = vc4.get_builtin qpu_id : index
// CHECK: %[[NUM0:.*]] = vc4.get_builtin num_qpus : index
// CHECK: %[[SRC:.*]] = vc4.get_uniform[0] : memref<?xf32>
// CHECK: %[[DST:.*]] = vc4.get_uniform[1] : memref<?xf32>
// CHECK: %[[SCALE:.*]] = vc4.get_uniform[2] : vector<16xf32>
// CHECK: %[[LIMIT:.*]] = vc4.get_uniform[3] : index
// CHECK: %[[C16_0:.*]] = arith.constant 16 : index
// CHECK: %[[BASE0:.*]] = arith.muli %[[QPU0]], %[[C16_0]] : index
// CHECK: %[[STRIDE0:.*]] = arith.muli %[[NUM0]], %[[C16_0]] : index
// CHECK: scf.for %[[IV0:.*]] = %[[BASE0]] to %[[LIMIT]] step %[[STRIDE0]] {
// CHECK:   vc4.dma_load %[[SRC]][%[[IV0]]] to slot[0] : memref<?xf32>, index
// CHECK:   %[[SRCV:.*]] = vc4.staged_read from slot[0] -> vector<16xf32>
// CHECK:   vc4.dma_load %[[DST]][%[[IV0]]] to slot[1] : memref<?xf32>, index
// CHECK:   %[[DSTV:.*]] = vc4.staged_read from slot[1] -> vector<16xf32>
// CHECK:   %[[MUL0:.*]] = vc4.fmul %[[SCALE]], %[[SRCV]] : vector<16xf32>, vector<16xf32> -> vector<16xf32>
// CHECK:   %[[ADD0:.*]] = vc4.fadd %[[MUL0]], %[[DSTV]] : vector<16xf32>, vector<16xf32> -> vector<16xf32>
// CHECK:   vc4.staged_write slot[2] = %[[ADD0]] : vector<16xf32>
// CHECK:   vc4.dma_store slot[2] to %[[DST]][%[[IV0]]] : memref<?xf32>, index
// CHECK: }

// CHECK: gpu.func @blend_linear
// CHECK: %[[QPU1:.*]] = vc4.get_builtin qpu_id : index
// CHECK: %[[NUM1:.*]] = vc4.get_builtin num_qpus : index
// CHECK: %[[LHS:.*]] = vc4.get_uniform[0] : memref<?xf32>
// CHECK: %[[RHS:.*]] = vc4.get_uniform[1] : memref<?xf32>
// CHECK: %[[OUT:.*]] = vc4.get_uniform[2] : memref<?xf32>
// CHECK: %[[ALPHA:.*]] = vc4.get_uniform[3] : vector<16xf32>
// CHECK: %[[BETA:.*]] = vc4.get_uniform[4] : vector<16xf32>
// CHECK: %[[EXTENT:.*]] = vc4.get_uniform[5] : index
// CHECK: %[[C16_1:.*]] = arith.constant 16 : index
// CHECK: %[[BASE1:.*]] = arith.muli %[[QPU1]], %[[C16_1]] : index
// CHECK: %[[STRIDE1:.*]] = arith.muli %[[NUM1]], %[[C16_1]] : index
// CHECK: scf.for %[[IV1:.*]] = %[[BASE1]] to %[[EXTENT]] step %[[STRIDE1]] {
// CHECK:   vc4.dma_load %[[LHS]][%[[IV1]]] to slot[0] : memref<?xf32>, index
// CHECK:   %[[LHSV:.*]] = vc4.staged_read from slot[0] -> vector<16xf32>
// CHECK:   vc4.dma_load %[[RHS]][%[[IV1]]] to slot[1] : memref<?xf32>, index
// CHECK:   %[[RHSV:.*]] = vc4.staged_read from slot[1] -> vector<16xf32>
// CHECK:   %[[LMUL:.*]] = vc4.fmul %[[ALPHA]], %[[LHSV]] : vector<16xf32>, vector<16xf32> -> vector<16xf32>
// CHECK:   %[[RMUL:.*]] = vc4.fmul %[[BETA]], %[[RHSV]] : vector<16xf32>, vector<16xf32> -> vector<16xf32>
// CHECK:   %[[SUM1:.*]] = vc4.fadd %[[LMUL]], %[[RMUL]] : vector<16xf32>, vector<16xf32> -> vector<16xf32>
// CHECK:   vc4.staged_write slot[2] = %[[SUM1]] : vector<16xf32>
// CHECK:   vc4.dma_store slot[2] to %[[OUT]][%[[IV1]]] : memref<?xf32>, index
// CHECK: }

// CHECK-NOT: cf.assert
