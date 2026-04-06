// RUN: vc4-opt %s | FileCheck %s

module {
  %row = arith.constant 0 : index
  %offset = arith.constant 0 : index
  %x = vc4.get_uniform 0 : memref<?xf32>
  %y = vc4.get_uniform 1 : memref<?xf32>

  "vc4.dma_load"(%x, %offset, %row) : (memref<?xf32>, index, index) -> ()
  %xv = vc4.vpm_read %row : index -> vector<16xf32>
  vc4.vpm_write %row, %xv : index, vector<16xf32>
  "vc4.dma_store"(%y, %offset, %row) : (memref<?xf32>, index, index) -> ()
}

// CHECK: module {
// CHECK:   %[[ROW:.*]] = arith.constant 0 : index
// CHECK:   %[[OFFSET:.*]] = arith.constant 0 : index
// CHECK:   %[[X:.*]] = vc4.get_uniform 0 : memref<?xf32>
// CHECK:   %[[Y:.*]] = vc4.get_uniform 1 : memref<?xf32>
// CHECK:   "vc4.dma_load"(%[[X]], %[[OFFSET]], %[[ROW]]) : (memref<?xf32>, index, index) -> ()
// CHECK:   %[[XV:.*]] = vc4.vpm_read %[[ROW]] : index -> vector<16xf32>
// CHECK:   vc4.vpm_write %[[ROW]], %[[XV]] : index, vector<16xf32>
// CHECK:   "vc4.dma_store"(%[[Y]], %[[OFFSET]], %[[ROW]]) : (memref<?xf32>, index, index) -> ()
// CHECK: }
