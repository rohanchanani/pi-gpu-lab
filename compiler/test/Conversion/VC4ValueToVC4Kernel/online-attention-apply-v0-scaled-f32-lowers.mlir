// RUN: vc4-opt %vc4_repo_root/compiler/test/ValueSurface/value-surface-online-attention-apply-scaled-valid.mlir --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

// CHECK-LABEL: vc4kernel.kernel @online_attention_apply_scaled
// CHECK: vc4kernel.splat
// CHECK: vc4kernel.fragment_alu.mul
// CHECK-SAME: opcode = #vc4kernel.mul_alu_opcode<fmul>
// CHECK: vc4kernel.fragment_reduce
// CHECK-SAME: kind = #vc4kernel.reduce<fmax>
// CHECK: vc4kernel.fragment_cmp
// CHECK: vc4kernel.fragment_sfu
// CHECK-SAME: kind = #vc4kernel.sfu_kind<exp>
// CHECK: vc4kernel.tmu_load_fragment
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: memref.load
// CHECK-NOT: math.
// CHECK-NOT: vector.
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
