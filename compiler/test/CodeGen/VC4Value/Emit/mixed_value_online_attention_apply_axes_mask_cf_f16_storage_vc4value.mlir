// RUN: rm -rf %t && mkdir -p %t
// RUN: sed 's/online_attention_f16_storage_inputs/mixed_value_online_attention_apply_axes_mask_cf_f16_storage_vc4value/g' %S/../../../ValueSurface/value-surface-online-attention-f16-storage-inputs-valid.mlir > %t/input.mlir
// RUN: vc4-opt %t/input.mlir --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel -o %t/mixed_value_online_attention.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VC4KERNEL --input-file=%t/mixed_value_online_attention.vc4kernel.mlir
// RUN: vc4-opt %t/mixed_value_online_attention.vc4kernel.mlir --verify-vc4kernel --convert-vc4kernel-to-ssavc4 -o %t/mixed_value_online_attention.ssavc4.mlir
// RUN: FileCheck %s --check-prefix=SSAVC4 --input-file=%t/mixed_value_online_attention.ssavc4.mlir
// RUN: vc4-opt %t/mixed_value_online_attention.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/mixed_value_online_attention.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/mixed_value_online_attention.vc4.mlir

// VC4KERNEL-LABEL: vc4kernel.kernel @mixed_value_online_attention_apply_axes_mask_cf_f16_storage_vc4value
// VC4KERNEL: vc4kernel.program_id
// VC4KERNEL: vc4kernel.program_id
// VC4KERNEL: cf.br
// VC4KERNEL: cf.cond_br
// VC4KERNEL: vc4kernel.vdr_load_rect_to_vpm
// VC4KERNEL: vc4kernel.fragment_unpack
// VC4KERNEL: vc4kernel.fragment_sfu
// VC4KERNEL: vc4kernel.vdw_store_fragment
// SSAVC4-LABEL: ssavc4.func @mixed_value_online_attention_apply_axes_mask_cf_f16_storage_vc4value
// SSAVC4: ssavc4.vdr.load
// SSAVC4: ssavc4.sfu
// SSAVC4: ssavc4.vdw.store
// VC4: vc4.module
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.
