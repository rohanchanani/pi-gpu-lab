// RUN: vc4-opt --lower-gpu-to-vc4 %s | FileCheck %s
// RUN: vc4-opt --lower-gpu-to-vc4 %s | vc4-translate --mlir-to-vc4asm | FileCheck %s --check-prefix=ASM
//
// Representative lowering test for the current feature-bounded subset.
// The source kernel happens to be SAXPY, but the backend contract being tested is:
// - logical gpu kernel input
// - explicit target builtins in lowered IR
// - user uniforms preserved in source signature order
// - persistent-worker/grid-stride execution
// - abstract staging slots rather than semantic VPM row numbers
// - no explicit tail/assert guard in the current lowered form
//
// ABI note:
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
// CHECK:     %[[C16:.*]] = arith.constant 16 : index
// CHECK:     %[[BASE:.*]] = arith.muli %[[QPU]], %[[C16]] : index
// CHECK:     %[[STRIDE:.*]] = arith.muli %[[NUM]], %[[C16]] : index
// CHECK:     scf.for %[[IV:.*]] = %[[BASE]] to %[[N]] step %[[STRIDE]] {
// CHECK:       vc4.dma_load %[[X]][%[[IV]]] to slot[0] : memref<?xf32>, index
// CHECK:       %[[XV:.*]] = vc4.staged_read from slot[0] -> vector<16xf32>
// CHECK:       vc4.dma_load %[[Y]][%[[IV]]] to slot[1] : memref<?xf32>, index
// CHECK:       %[[YV:.*]] = vc4.staged_read from slot[1] -> vector<16xf32>
// CHECK:       %[[MUL:.*]] = vc4.fmul %[[A]], %[[XV]] : vector<16xf32>, vector<16xf32> -> vector<16xf32>
// CHECK:       %[[SUM:.*]] = vc4.fadd %[[MUL]], %[[YV]] : vector<16xf32>, vector<16xf32> -> vector<16xf32>
// CHECK:       vc4.staged_write slot[2] = %[[SUM]] : vector<16xf32>
// CHECK:       vc4.dma_store slot[2] to %[[Y]][%[[IV]]] : memref<?xf32>, index
// CHECK:     }
// CHECK:     gpu.return
// CHECK:   }
// CHECK: }
// CHECK-NOT: cf.assert
// CHECK-NOT: vc4.vpm_read
// CHECK-NOT: vc4.vpm_write

// ASM: ; inspection-only VC4 assembly sketch generated from lowered vc4 IR
// ASM: ; gpu.module @kernels
// ASM: ; gpu.func @saxpy
// ASM: ; Uniform reads
// ASM: ; uniform_0 = vc4.get_uniform[0] : memref<?xf32>
// ASM: mov uniform_0, unif
// ASM: ; uniform_1 = vc4.get_uniform[1] : memref<?xf32>
// ASM: mov uniform_1, unif
// ASM: ; uniform_2 = vc4.get_uniform[2] : vector<16xf32>
// ASM: mov uniform_2, unif
// ASM: ; uniform_3 = vc4.get_uniform[3] : index
// ASM: mov uniform_3, unif
// ASM: ; Builtin suffix reads
// ASM: ; builtin_qpu_id = vc4.get_builtin qpu_id : index
// ASM: mov builtin_qpu_id, unif
// ASM: ; builtin_num_qpus = vc4.get_builtin num_qpus : index
// ASM: mov builtin_num_qpus, unif
// ASM: ; Scalar/index setup
// ASM: ; base = builtin_qpu_id * 16
// ASM: ; stride = builtin_num_qpus * 16
// ASM: ; Control
// ASM: ; scf.for iv0 = base to uniform_3 step stride
// ASM: ; DMA staging
// ASM: ; vc4.dma_load uniform_0[iv0] to slot[0]
// ASM: ; debug-only emitter-local mapping: slot[0] currently prints as physical row
// ASM: ; Staging access
// ASM: ; tmp0 = vc4.staged_read from slot[0]
// ASM: ; DMA staging
// ASM: ; vc4.dma_load uniform_1[iv0] to slot[1]
// ASM: ; debug-only emitter-local mapping: slot[1] currently prints as physical row
// ASM: ; Staging access
// ASM: ; tmp1 = vc4.staged_read from slot[1]
// ASM: ; Compute
// ASM: ; tmp2 = vc4.fmul uniform_2, tmp0
// ASM: fmul tmp2, uniform_2, tmp0
// ASM: ; tmp3 = vc4.fadd tmp2, tmp1
// ASM: fadd tmp3, tmp2, tmp1
// ASM: ; Staging access
// ASM: ; vc4.staged_write slot[2] = tmp3
// ASM: ; debug-only emitter-local mapping: slot[2] currently prints as physical row
// ASM: ; DMA staging
// ASM: ; vc4.dma_store slot[2] to uniform_1[iv0]
// ASM: ; debug-only emitter-local mapping: slot[2] currently prints as physical row
