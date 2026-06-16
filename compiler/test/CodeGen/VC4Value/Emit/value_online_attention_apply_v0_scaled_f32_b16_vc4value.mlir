// RUN: rm -rf %t && mkdir -p %t
// RUN: sed 's/online_attention_apply_scaled/value_online_attention_apply_v0_scaled_f32_b16_vc4value/g' %S/../../../ValueSurface/value-surface-online-attention-apply-scaled-valid.mlir > %t/input.mlir
// RUN: vc4-opt %t/input.mlir --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel -o %t/value_online_attention_scaled.vc4kernel.mlir
// RUN: FileCheck %s --check-prefix=VC4KERNEL --input-file=%t/value_online_attention_scaled.vc4kernel.mlir
// RUN: vc4-opt %t/value_online_attention_scaled.vc4kernel.mlir --verify-vc4kernel --convert-vc4kernel-to-ssavc4 -o %t/value_online_attention_scaled.ssavc4.mlir
// RUN: vc4-opt %t/value_online_attention_scaled.ssavc4.mlir --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses -o %t/value_online_attention_scaled.vc4.mlir
// RUN: FileCheck %s --check-prefix=VC4 --input-file=%t/value_online_attention_scaled.vc4.mlir

// VC4KERNEL-LABEL: vc4kernel.kernel @value_online_attention_apply_v0_scaled_f32_b16_vc4value
// VC4KERNEL: vc4kernel.splat
// VC4KERNEL: vc4kernel.fragment_alu.mul
// VC4KERNEL-SAME: opcode = #vc4kernel.mul_alu_opcode<fmul>
// VC4KERNEL: vc4kernel.fragment_sfu
// VC4KERNEL-SAME: kind = #vc4kernel.sfu_kind<exp>
// VC4KERNEL: vc4kernel.vdw_store_fragment
// VC4: vc4.module
// VC4-NOT: ssavc4.
// VC4-NOT: vc4kernel.
