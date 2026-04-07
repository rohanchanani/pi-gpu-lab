// RUN: vc4-opt --lower-gpu-saxpy-to-vc4 %s | FileCheck %s
// RUN: vc4-opt --lower-gpu-saxpy-to-vc4 %s | vc4-translate --mlir-to-vc4asm | FileCheck %s --check-prefix=ASM
//
// ABI note for this authoritative lowering test:
// - source kernel ABI stays (%x: memref<?xf32>, %y: memref<?xf32>, %a: f32, %n: index)
// - lowered IR keeps execution builtins distinct from user uniforms
// - lowered alpha is materialized as vc4.get_uniform[2] : vector<16xf32>
//   to reflect current VC4 lane-broadcast uniform semantics
// - prototype runtime uniform payload order is:
//   [0] x, [1] y, [2] scalar a, [3] n, [4] qpu_id, [5] num_qpus
// - final physical kernel realization must consume that payload sequentially
//   with repeated mov ..., unif; it is not random-access storage

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

// CHECK: gpu.module @kernels {
// CHECK:   gpu.func @saxpy(%[[XARG:.*]]: memref<?xf32>, %[[YARG:.*]]: memref<?xf32>, %[[AARG:.*]]: f32, %[[NARG:.*]]: index) kernel {
// CHECK:     %[[QPU:.*]] = vc4.get_builtin qpu_id : index
// CHECK:     %[[NUM:.*]] = vc4.get_builtin num_qpus : index
// CHECK:     %[[X:.*]] = vc4.get_uniform[0] : memref<?xf32>
// CHECK:     %[[Y:.*]] = vc4.get_uniform[1] : memref<?xf32>
// CHECK:     %[[A:.*]] = vc4.get_uniform[2] : vector<16xf32>
// CHECK:     %[[N:.*]] = vc4.get_uniform[3] : index
// CHECK:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:     %[[C16:.*]] = arith.constant 16 : index
// CHECK:     %[[BASE:.*]] = arith.muli %[[QPU]], %[[C16]] : index
// CHECK:     %[[STRIDE:.*]] = arith.muli %[[NUM]], %[[C16]] : index
// CHECK:     %[[REM:.*]] = arith.remui %[[N]], %[[C16]] : index
// CHECK:     %[[OK:.*]] = arith.cmpi eq, %[[REM]], %[[C0]] : index
// CHECK:     cf.assert %[[OK]], "lower-gpu-saxpy-to-vc4 requires n to be a multiple of 16"
// CHECK:     scf.for %[[IV:.*]] = %[[BASE]] to %[[N]] step %[[STRIDE]] {
// CHECK:       vc4.dma_load %[[X]][%[[IV]]] to %[[C0]] : memref<?xf32>, index, index
// CHECK:       vc4.dma_load %[[Y]][%[[IV]]] to %[[C1]] : memref<?xf32>, index, index
// CHECK:       %[[XV:.*]] = vc4.vpm_read from %[[C0]] : index -> vector<16xf32>
// CHECK:       %[[YV:.*]] = vc4.vpm_read from %[[C1]] : index -> vector<16xf32>
// CHECK:       %[[MUL:.*]] = vc4.fmul %[[A]], %[[XV]] : vector<16xf32>, vector<16xf32> -> vector<16xf32>
// CHECK:       %[[SUM:.*]] = vc4.fadd %[[MUL]], %[[YV]] : vector<16xf32>, vector<16xf32> -> vector<16xf32>
// CHECK:       vc4.vpm_write %[[SUM]] to %[[C1]] : vector<16xf32>, index
// CHECK:       vc4.dma_store %[[C1]] to %[[Y]][%[[IV]]] : index, memref<?xf32>, index
// CHECK:     }
// CHECK:     gpu.return
// CHECK:   }
// CHECK: }

// ASM: ; gpu.module @kernels
// ASM: ; gpu.func @saxpy
// ASM: ; Execution builtins
// ASM: ; builtin_qpu_id = vc4.get_builtin qpu_id : index
// ASM: mov builtin_qpu_id, qpu_id
// ASM: ; builtin_num_qpus = vc4.get_builtin num_qpus : index
// ASM: mov builtin_num_qpus, num_qpus
// ASM: ; Uniform reads
// ASM: ; uniform_0 = vc4.get_uniform[0] : memref<?xf32>
// ASM: ; uniform_1 = vc4.get_uniform[1] : memref<?xf32>
// ASM: ; uniform_2 = vc4.get_uniform[2] : vector<16xf32>
// ASM: ; uniform_3 = vc4.get_uniform[3] : index
// ASM: ; Scalar/index setup
// ASM: ; tmp0 = arith.muli builtin_qpu_id, 16
// ASM: ; tmp1 = arith.muli builtin_num_qpus, 16
// ASM: ; tmp2 = arith.remui uniform_3, 16
// ASM: ; tmp3 = arith.cmpi eq, tmp2, 0
// ASM: ; Control
// ASM: ; cf.assert tmp3, "lower-gpu-saxpy-to-vc4 requires n to be a multiple of 16"
// ASM: ; scf.for iv0 = tmp0 to uniform_3 step tmp1
// ASM: ; DMA staging
// ASM: ; vc4.dma_load uniform_0[iv0] to row 0
// ASM: ; vc4.dma_load uniform_1[iv0] to row 1
// ASM: ; VPM access
// ASM: ; tmp4 = vc4.vpm_read from row 0
// ASM: ; tmp5 = vc4.vpm_read from row 1
// ASM: ; Compute
// ASM: ; tmp6 = vc4.fmul uniform_2, tmp4
// ASM: ; tmp7 = vc4.fadd tmp6, tmp5
// ASM: ; VPM access
// ASM: ; vc4.vpm_write tmp7 to row 1
// ASM: ; DMA staging
// ASM: ; vc4.dma_store row 1 to uniform_1[iv0]
// ASM: ; end scf.for iv0
