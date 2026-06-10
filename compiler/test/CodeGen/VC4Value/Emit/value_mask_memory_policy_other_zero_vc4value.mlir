// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel -o %t/value_mask_memory_policy_other_zero.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VC4KERNEL --input-file=%t/value_mask_memory_policy_other_zero.vc4kernel.mlir
// RUN: vc4-opt %t/value_mask_memory_policy_other_zero.vc4kernel.mlir --verify-vc4kernel --convert-vc4kernel-to-ssavc4 -o %t/value_mask_memory_policy_other_zero.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=SSAVC4 --input-file=%t/value_mask_memory_policy_other_zero.ssavc4.mlir
// RUN: vc4-opt %t/value_mask_memory_policy_other_zero.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/value_mask_memory_policy_other_zero.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/value_mask_memory_policy_other_zero.vc4.mlir

func.func @value_mask_memory_policy_other_zero_vc4value(
    %in: memref<64xf32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %out: memref<64xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %base: index {vc4value.arg_name = "base"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 1 : i32} {
  %zero = arith.constant 0.000000e+00 : f32
  %tail = vector.create_mask %n : vector<16xi1>
  %x = vector.transfer_read %in[%base], %zero, %tail {in_bounds = [true]} : memref<64xf32, #vc4value.global>, vector<16xf32>
  vector.transfer_write %x, %out[%base], %tail {in_bounds = [true]} : vector<16xf32>, memref<64xf32, #vc4value.global>
  return
}

// VC4KERNEL-LABEL: vc4kernel.kernel @value_mask_memory_policy_other_zero_vc4value
// VC4KERNEL: vc4kernel.tmu_load_fragment
// VC4KERNEL-SAME: inactive_load = #vc4kernel.inactive_load<zero>
// VC4KERNEL: vc4kernel.vdw_store_fragment
// VC4KERNEL-SAME: inactive_store = #vc4kernel.inactive_store<preserve>
// VC4KERNEL-NOT: func.func
// VC4KERNEL-NOT: vc4value.
// VC4KERNEL-NOT: vector.
// VC4KERNEL-NOT: memref.

// SSAVC4-LABEL: ssavc4.func @value_mask_memory_policy_other_zero_vc4value
// SSAVC4: ssavc4.tmu.request
// SSAVC4: ssavc4.vdw.store

// VC4: vc4.module
// VC4: vc4.qpu.
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.
