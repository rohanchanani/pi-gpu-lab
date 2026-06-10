// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel -o %t/mixed_value_mask_memory_axis_cf.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VC4KERNEL --input-file=%t/mixed_value_mask_memory_axis_cf.vc4kernel.mlir
// RUN: vc4-opt %t/mixed_value_mask_memory_axis_cf.vc4kernel.mlir --verify-vc4kernel --convert-vc4kernel-to-ssavc4 -o %t/mixed_value_mask_memory_axis_cf.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=SSAVC4 --input-file=%t/mixed_value_mask_memory_axis_cf.ssavc4.mlir
// RUN: vc4-opt %t/mixed_value_mask_memory_axis_cf.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/mixed_value_mask_memory_axis_cf.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/mixed_value_mask_memory_axis_cf.vc4.mlir

func.func @mixed_value_mask_memory_axis_cf_vc4value(
    %in: memref<256xi32, #vc4value.global> {vc4value.arg_name = "in", vc4value.direction = "in"},
    %out: memref<256xi32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out"},
    %n: index {vc4value.arg_name = "n"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32} {
  %pid0 = vc4value.program_id {axis = 0 : i32} : index
  %pid1 = vc4value.program_id {axis = 1 : i32} : index
  %num0 = vc4value.num_programs {axis = 0 : i32} : index
  %block_row = arith.muli %pid1, %num0 : index
  %block_id = arith.addi %block_row, %pid0 : index
  %c16 = arith.constant 16 : index
  %base = arith.muli %block_id, %c16 : index
  %tail_count = arith.subi %n, %base : index
  %tail = vector.create_mask %tail_count : vector<16xi1>
  %zero = arith.constant 0 : i32
  %x = vector.transfer_read %in[%base], %zero, %tail {in_bounds = [true]} : memref<256xi32, #vc4value.global>, vector<16xi32>
  %take = arith.cmpi ule, %pid0, %pid1 : index
  cf.cond_br %take, ^store, ^done
^store:
  vector.transfer_write %x, %out[%base], %tail {in_bounds = [true]} : vector<16xi32>, memref<256xi32, #vc4value.global>
  cf.br ^done
^done:
  return
}

// VC4KERNEL-LABEL: vc4kernel.kernel @mixed_value_mask_memory_axis_cf_vc4value
// VC4KERNEL: vc4kernel.program_id
// VC4KERNEL-SAME: axis = 0
// VC4KERNEL: vc4kernel.program_id
// VC4KERNEL-SAME: axis = 1
// VC4KERNEL: vc4kernel.num_programs
// VC4KERNEL: vc4kernel.pred.tail
// VC4KERNEL: vc4kernel.tmu_load_fragment
// VC4KERNEL: cf.cond_br
// VC4KERNEL: vc4kernel.vdw_store_fragment
// VC4KERNEL-NOT: func.func
// VC4KERNEL-NOT: vc4value.
// VC4KERNEL-NOT: vector.
// VC4KERNEL-NOT: memref.

// SSAVC4-LABEL: ssavc4.func @mixed_value_mask_memory_axis_cf_vc4value
// SSAVC4: program_id_x
// SSAVC4: program_id_y
// SSAVC4: ssavc4.tmu.request
// SSAVC4: ssavc4.vdw.store

// VC4: vc4.module
// VC4: vc4.qpu.
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.
