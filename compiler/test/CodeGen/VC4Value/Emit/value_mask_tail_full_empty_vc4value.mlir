// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel -o %t/value_mask_tail_full_empty.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VC4KERNEL --input-file=%t/value_mask_tail_full_empty.vc4kernel.mlir
// RUN: vc4-opt %t/value_mask_tail_full_empty.vc4kernel.mlir --verify-vc4kernel --convert-vc4kernel-to-ssavc4 -o %t/value_mask_tail_full_empty.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=SSAVC4 --input-file=%t/value_mask_tail_full_empty.ssavc4.mlir
// RUN: vc4-opt %t/value_mask_tail_full_empty.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/value_mask_tail_full_empty.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/value_mask_tail_full_empty.vc4.mlir

func.func @value_mask_tail_full_empty_vc4value(
    %in: memref<64xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %out: memref<64xi32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %base: index {vc4value.arg_name = "base"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %zero_i32 = arith.constant 0 : i32
  %zero_idx = arith.constant 0 : index
  %tail = vector.create_mask %n : vector<16xi1>
  %empty = vector.create_mask %zero_idx : vector<16xi1>
  %v = vector.transfer_read %in[%base], %zero_i32 {in_bounds = [true]} : memref<64xi32, #vc4value.global>, vector<16xi32>
  vector.transfer_write %v, %out[%base], %tail {in_bounds = [true]} : vector<16xi32>, memref<64xi32, #vc4value.global>
  vector.transfer_write %v, %out[%base], %empty {in_bounds = [true]} : vector<16xi32>, memref<64xi32, #vc4value.global>
  return
}

// VC4KERNEL-LABEL: vc4kernel.kernel @value_mask_tail_full_empty_vc4value
// VC4KERNEL: vc4kernel.pred.tail
// VC4KERNEL: vc4kernel.pred.empty
// VC4KERNEL: vc4kernel.pred.full
// VC4KERNEL: vc4kernel.tmu_load_fragment
// VC4KERNEL: vc4kernel.vdw_store_fragment
// VC4KERNEL: vc4kernel.vdw_store_fragment
// VC4KERNEL-NOT: func.func
// VC4KERNEL-NOT: vc4value.
// VC4KERNEL-NOT: vector.
// VC4KERNEL-NOT: memref.

// SSAVC4-LABEL: ssavc4.func @value_mask_tail_full_empty_vc4value
// SSAVC4: ssavc4.tmu.request
// SSAVC4: ssavc4.vdw.store
// SSAVC4-NOT: vc4kernel.

// VC4: vc4.module
// VC4: vc4.qpu.
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.
