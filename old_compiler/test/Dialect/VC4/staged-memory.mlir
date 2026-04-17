// Dialect smoke test for abstract staged-memory VC4 ops plus vector compute.
// RUN: vc4-opt %s | FileCheck %s

module {
  %offset = arith.constant 0 : index
  %x = vc4.get_uniform[0] : memref<?xf32>
  %y = vc4.get_uniform[1] : memref<?xf32>
  %a = vc4.get_uniform[2] : vector<16xf32>

  vc4.dma_load %x[%offset] to slot[2] : memref<?xf32>, index
  vc4.dma_load %y[%offset] to slot[3] : memref<?xf32>, index
  %xv = vc4.staged_read from slot[2] -> vector<16xf32>
  %yv = vc4.staged_read from slot[3] -> vector<16xf32>
  %mul = vc4.fmul %a, %xv : vector<16xf32>, vector<16xf32> -> vector<16xf32>
  %sum = vc4.fadd %mul, %yv : vector<16xf32>, vector<16xf32> -> vector<16xf32>
  vc4.staged_write slot[5] = %sum : vector<16xf32>
  vc4.dma_store slot[5] to %y[%offset] : memref<?xf32>, index
}

// CHECK: module {
// CHECK:   %[[OFFSET:.*]] = arith.constant 0 : index
// CHECK:   %[[X:.*]] = vc4.get_uniform[0] : memref<?xf32>
// CHECK:   %[[Y:.*]] = vc4.get_uniform[1] : memref<?xf32>
// CHECK:   %[[A:.*]] = vc4.get_uniform[2] : vector<16xf32>
// CHECK:   vc4.dma_load %[[X]][%[[OFFSET]]] to slot[2] : memref<?xf32>, index
// CHECK:   vc4.dma_load %[[Y]][%[[OFFSET]]] to slot[3] : memref<?xf32>, index
// CHECK:   %[[XV:.*]] = vc4.staged_read from slot[2] -> vector<16xf32>
// CHECK:   %[[YV:.*]] = vc4.staged_read from slot[3] -> vector<16xf32>
// CHECK:   %[[MUL:.*]] = vc4.fmul %[[A]], %[[XV]] : vector<16xf32>, vector<16xf32> -> vector<16xf32>
// CHECK:   %[[SUM:.*]] = vc4.fadd %[[MUL]], %[[YV]] : vector<16xf32>, vector<16xf32> -> vector<16xf32>
// CHECK:   vc4.staged_write slot[5] = %[[SUM]] : vector<16xf32>
// CHECK:   vc4.dma_store slot[5] to %[[Y]][%[[OFFSET]]] : memref<?xf32>, index
// CHECK: }
