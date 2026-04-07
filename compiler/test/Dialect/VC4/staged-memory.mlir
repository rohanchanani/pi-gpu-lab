// Dialect smoke test for staged-memory VC4 ops plus vector compute.
// RUN: vc4-opt %s | FileCheck %s

module {
  %xrow = arith.constant 0 : index
  %yrow = arith.constant 1 : index
  %offset = arith.constant 0 : index
  %x = vc4.get_uniform[0] : memref<?xf32>
  %y = vc4.get_uniform[1] : memref<?xf32>
  %a = vc4.get_uniform[2] : vector<16xf32>

  vc4.dma_load %x[%offset] to %xrow : memref<?xf32>, index, index
  vc4.dma_load %y[%offset] to %yrow : memref<?xf32>, index, index
  %xv = vc4.vpm_read from %xrow : index -> vector<16xf32>
  %yv = vc4.vpm_read from %yrow : index -> vector<16xf32>
  %mul = vc4.fmul %a, %xv : vector<16xf32>, vector<16xf32> -> vector<16xf32>
  %sum = vc4.fadd %mul, %yv : vector<16xf32>, vector<16xf32> -> vector<16xf32>
  vc4.vpm_write %sum to %yrow : vector<16xf32>, index
  vc4.dma_store %yrow to %y[%offset] : index, memref<?xf32>, index
}

// CHECK: module {
// CHECK:   %[[XROW:.*]] = arith.constant 0 : index
// CHECK:   %[[YROW:.*]] = arith.constant 1 : index
// CHECK:   %[[OFFSET:.*]] = arith.constant 0 : index
// CHECK:   %[[X:.*]] = vc4.get_uniform[0] : memref<?xf32>
// CHECK:   %[[Y:.*]] = vc4.get_uniform[1] : memref<?xf32>
// CHECK:   %[[A:.*]] = vc4.get_uniform[2] : vector<16xf32>
// CHECK:   vc4.dma_load %[[X]][%[[OFFSET]]] to %[[XROW]] : memref<?xf32>, index, index
// CHECK:   vc4.dma_load %[[Y]][%[[OFFSET]]] to %[[YROW]] : memref<?xf32>, index, index
// CHECK:   %[[XV:.*]] = vc4.vpm_read from %[[XROW]] : index -> vector<16xf32>
// CHECK:   %[[YV:.*]] = vc4.vpm_read from %[[YROW]] : index -> vector<16xf32>
// CHECK:   %[[MUL:.*]] = vc4.fmul %[[A]], %[[XV]] : vector<16xf32>, vector<16xf32> -> vector<16xf32>
// CHECK:   %[[SUM:.*]] = vc4.fadd %[[MUL]], %[[YV]] : vector<16xf32>, vector<16xf32> -> vector<16xf32>
// CHECK:   vc4.vpm_write %[[SUM]] to %[[YROW]] : vector<16xf32>, index
// CHECK:   vc4.dma_store %[[YROW]] to %[[Y]][%[[OFFSET]]] : index, memref<?xf32>, index
// CHECK: }
