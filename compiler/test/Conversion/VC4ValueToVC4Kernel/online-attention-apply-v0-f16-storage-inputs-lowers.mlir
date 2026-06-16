// RUN: vc4-opt %vc4_repo_root/compiler/test/ValueSurface/value-surface-online-attention-f16-storage-inputs-valid.mlir --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

// CHECK-LABEL: vc4kernel.kernel @online_attention_f16_storage_inputs
// CHECK: vc4kernel.vdr_load_rect_to_vpm
// CHECK: vc4kernel.vpm_read_fragment
// CHECK: vc4kernel.fragment_unpack
// CHECK-SAME: source = #vc4kernel.subword_type<f16>
// CHECK: vc4kernel.fragment_reduce
// CHECK-SAME: kind = #vc4kernel.reduce<fmax>
// CHECK: vc4kernel.fragment_sfu
// CHECK-SAME: kind = #vc4kernel.sfu_kind<exp>
// CHECK: vc4kernel.vdr_load_rect_to_vpm
// CHECK: vc4kernel.vpm_read_fragment
// CHECK: vc4kernel.fragment_unpack
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: math.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
