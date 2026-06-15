// RUN: rm -rf %t && mkdir -p %t
// RUN: vc4-opt %s --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel -o %t/mixed_value_attention_apply_v0.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VC4KERNEL --input-file=%t/mixed_value_attention_apply_v0.vc4kernel.mlir
// RUN: vc4-opt %t/mixed_value_attention_apply_v0.vc4kernel.mlir --verify-vc4kernel --convert-vc4kernel-to-ssavc4 -o %t/mixed_value_attention_apply_v0.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=SSAVC4 --input-file=%t/mixed_value_attention_apply_v0.ssavc4.mlir
// RUN: vc4-opt %t/mixed_value_attention_apply_v0.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/mixed_value_attention_apply_v0.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/mixed_value_attention_apply_v0.vc4.mlir

func.func @mixed_value_attention_apply_v0_axes_mask_cf_f16_storage_vc4value(
    %scores: memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "scores", vc4value.direction = "in", vc4value.shape_args = ["queries", "kdim"], vc4value.stride_args = ["lds"]},
    %vt: memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global> {vc4value.arg_name = "vt", vc4value.direction = "in", vc4value.shape_args = ["dims", "kdim"], vc4value.stride_args = ["ldv"]},
    %out: memref<?xf32, #vc4value.global> {vc4value.arg_name = "out", vc4value.direction = "out", vc4value.shape_args = ["out_n"]},
    %audit: memref<?xf32, #vc4value.global> {vc4value.arg_name = "audit", vc4value.direction = "out", vc4value.shape_args = ["queries"]},
    %k: index {vc4value.arg_name = "k", vc4value.scalar_role = "extent"},
    %queries: index {vc4value.arg_name = "queries", vc4value.scalar_role = "extent"},
    %kdim: index {vc4value.arg_name = "kdim", vc4value.scalar_role = "extent"},
    %dims: index {vc4value.arg_name = "dims", vc4value.scalar_role = "extent"},
    %out_n: index {vc4value.arg_name = "out_n", vc4value.scalar_role = "extent"},
    %lds: index {vc4value.arg_name = "lds", vc4value.scalar_role = "stride"},
    %ldv: index {vc4value.arg_name = "ldv", vc4value.scalar_role = "stride"},
    %ldo: index {vc4value.arg_name = "ldo", vc4value.scalar_role = "stride"},
    %scale: f32 {vc4value.arg_name = "scale", vc4value.scalar_role = "value"})
    attributes {vc4value.kernel, vc4value.grid_rank = 3 : i32,
                vc4value.attention_apply_v0 = "precomputed_transposed_v_active_1_to_16",
                vc4value.softmax_v0 = "one_block_active_1_to_16",
                vc4value.math_policy = "approx_sfu",
                vc4value.fp_domain = "finite",
                vc4value.reduction_policy = "finite_tree",
                vc4value.max_policy = "finite",
                vc4value.f16_storage_policy = "finite"} {
  %q = vc4value.program_id {axis = 0 : i32} : index
  %d = vc4value.program_id {axis = 1 : i32} : index
  %group = vc4value.program_id {axis = 2 : i32} : index
  %c0 = arith.constant 0 : index
  %is_group0 = arith.cmpi eq, %group, %c0 : index
  cf.cond_br %is_group0, ^body, ^done
^body:
  %obase = arith.muli %q, %ldo : index
  %oidx = arith.addi %obase, %d : index
  %mask = vector.create_mask %k : vector<16xi1>
  %zero_h = arith.constant 0.000000e+00 : f16
  %scores_h = vector.transfer_read %scores[%q, %c0], %zero_h, %mask {in_bounds = [true]} : memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf16>
  %scores_f = arith.extf %scores_h : vector<16xf16> to vector<16xf32>
  %scale_v = vector.broadcast %scale : f32 to vector<16xf32>
  %scaled = arith.mulf %scores_f, %scale_v : vector<16xf32>
  %low = arith.constant dense<-8.000000e+01> : vector<16xf32>
  %zeros = arith.constant dense<0.000000e+00> : vector<16xf32>
  %active_scores = arith.select %mask, %scaled, %low : vector<16xi1>, vector<16xf32>
  %max = vector.reduction <maxnumf>, %active_scores : vector<16xf32> into f32
  %maxv = vector.broadcast %max : f32 to vector<16xf32>
  %shifted = arith.subf %active_scores, %maxv : vector<16xf32>
  %e = math.exp %shifted : vector<16xf32>
  %active_e = arith.select %mask, %e, %zeros : vector<16xi1>, vector<16xf32>
  %denom = vector.reduction <add>, %active_e : vector<16xf32> into f32
  %one = arith.constant 1.000000e+00 : f32
  %inv = arith.divf %one, %denom : f32
  %invv = vector.broadcast %inv : f32 to vector<16xf32>
  %probs = arith.mulf %active_e, %invv : vector<16xf32>
  %vh = vector.transfer_read %vt[%d, %c0], %zero_h, %mask {in_bounds = [true]} : memref<?x?xf16, strided<[?, 1], offset: 0>, #vc4value.global>, vector<16xf16>
  %v = arith.extf %vh : vector<16xf16> to vector<16xf32>
  %weighted = arith.mulf %probs, %v : vector<16xf32>
  %acc = vector.reduction <add>, %weighted : vector<16xf32> into f32
  memref.store %acc, %out[%oidx] : memref<?xf32, #vc4value.global>
  memref.store %denom, %audit[%q] : memref<?xf32, #vc4value.global>
  cf.br ^done
^done:
  return
}

// VC4KERNEL-LABEL: vc4kernel.kernel @mixed_value_attention_apply_v0_axes_mask_cf_f16_storage_vc4value
// VC4KERNEL: vc4kernel.program_id {{.*}}axis = 2
// VC4KERNEL: cf.cond_br
// VC4KERNEL: vc4kernel.vdr_load_rect_to_vpm
// VC4KERNEL: vc4kernel.fragment_unpack
// VC4KERNEL: vc4kernel.fragment_sfu
// VC4KERNEL-SAME: kind = #vc4kernel.sfu_kind<exp>
// VC4KERNEL: vc4kernel.vdr_load_rect_to_vpm
// VC4KERNEL: vc4kernel.fragment_unpack
// VC4KERNEL: vc4kernel.vdw_store_fragment
// VC4KERNEL: vc4kernel.vdw_store_fragment
// SSAVC4-LABEL: ssavc4.func @mixed_value_attention_apply_v0_axes_mask_cf_f16_storage_vc4value
// SSAVC4: ssavc4.sfu
// SSAVC4: f16_storage_conversion
// VC4: vc4.module
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.
