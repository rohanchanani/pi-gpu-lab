// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel -o %t/mixed_value_sfu_softmax.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VC4KERNEL --input-file=%t/mixed_value_sfu_softmax.vc4kernel.mlir
// RUN: vc4-opt %t/mixed_value_sfu_softmax.vc4kernel.mlir --verify-vc4kernel --convert-vc4kernel-to-ssavc4 -o %t/mixed_value_sfu_softmax.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=SSAVC4 --input-file=%t/mixed_value_sfu_softmax.ssavc4.mlir
// RUN: vc4-opt %t/mixed_value_sfu_softmax.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/mixed_value_sfu_softmax.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/mixed_value_sfu_softmax.vc4.mlir

func.func @mixed_value_sfu_softmax_axes_mask_cf_f16_storage_vc4value(
    %x: memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "x", vc4value.direction = "in", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["stride"]},
    %y: memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "y", vc4value.direction = "out", vc4value.shape_args = ["rows", "cols"], vc4value.stride_args = ["stride"]},
    %rows: index {vc4value.arg_name = "rows", vc4value.scalar_role = "extent"},
    %cols: index {vc4value.arg_name = "cols", vc4value.scalar_role = "extent"},
    %stride: index {vc4value.arg_name = "stride", vc4value.scalar_role = "stride"})
    attributes {vc4value.kernel, vc4value.grid_rank = 2 : i32,
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree",
                vc4value.max_policy = "finite",
                vc4value.f16_storage_policy = "finite",
                vc4value.softmax_v0 = "one_block_active_1_to_16"} {
  %block = vc4value.program_id {axis = 0 : i32} : index
  %row = vc4value.program_id {axis = 1 : i32} : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %col = arith.muli %block, %c16 : index
  %is_first = arith.cmpi eq, %block, %c0 : index
  cf.cond_br %is_first, ^body, ^done
^body:
  %mask = vector.create_mask %cols : vector<16xi1>
  %zero_h = arith.constant 0.000000e+00 : f16
  %xh = vector.transfer_read %x[%row, %col], %zero_h, %mask {in_bounds = [true]} : memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf16>
  %xf = arith.extf %xh : vector<16xf16> to vector<16xf32>
  %low = arith.constant dense<-8.000000e+01> : vector<16xf32>
  %zero = arith.constant dense<0.000000e+00> : vector<16xf32>
  %active_x = arith.select %mask, %xf, %low : vector<16xi1>, vector<16xf32>
  %max = vector.reduction <maxnumf>, %active_x : vector<16xf32> into f32
  %maxv = vector.broadcast %max : f32 to vector<16xf32>
  %shifted = arith.subf %xf, %maxv : vector<16xf32>
  %expv = math.exp %shifted : vector<16xf32>
  %active_e = arith.select %mask, %expv, %zero : vector<16xi1>, vector<16xf32>
  %denom = vector.reduction <add>, %active_e : vector<16xf32> into f32
  %one = arith.constant 1.000000e+00 : f32
  %inv = arith.divf %one, %denom : f32
  %invv = vector.broadcast %inv : f32 to vector<16xf32>
  %outf = arith.mulf %active_e, %invv : vector<16xf32>
  %outh = arith.truncf %outf : vector<16xf32> to vector<16xf16>
  vector.transfer_write %outh, %y[%row, %col], %mask {in_bounds = [true]} : vector<16xf16>, memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global>
  cf.br ^done
^done:
  return
}

// VC4KERNEL-LABEL: vc4kernel.kernel @mixed_value_sfu_softmax_axes_mask_cf_f16_storage_vc4value
// VC4KERNEL: vc4kernel.program_id
// VC4KERNEL-SAME: axis = 0
// VC4KERNEL: vc4kernel.program_id
// VC4KERNEL-SAME: axis = 1
// VC4KERNEL: cf.cond_br
// VC4KERNEL: vc4kernel.fragment_unpack
// VC4KERNEL: vc4kernel.fragment_reduce
// VC4KERNEL-SAME: kind = #vc4kernel.reduce<fmax>
// VC4KERNEL: vc4kernel.fragment_const {{.*}}1.442695
// VC4KERNEL: vc4kernel.fragment_alu.mul
// VC4KERNEL-SAME: opcode = #vc4kernel.mul_alu_opcode<fmul>
// VC4KERNEL: vc4kernel.fragment_sfu
// VC4KERNEL-SAME: kind = #vc4kernel.sfu_kind<exp>
// VC4KERNEL: vc4kernel.fragment_sfu
// VC4KERNEL-SAME: kind = #vc4kernel.sfu_kind<recip>
// VC4KERNEL: vc4kernel.fragment_pack
// SSAVC4-LABEL: ssavc4.func @mixed_value_sfu_softmax_axes_mask_cf_f16_storage_vc4value
// SSAVC4: ssavc4.sfu
// SSAVC4: f16_storage_conversion
// VC4: vc4.module
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.
