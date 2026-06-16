// RUN: vc4-opt %vc4_repo_root/compiler/test/ValueSurface/value-surface-online-attention-apply-v0-composite-valid.mlir --convert-scf-to-cf --vc4-verify-value-surface --convert-vc4-value-to-vc4kernel --verify-vc4kernel | FileCheck %s

// CHECK-LABEL: vc4kernel.kernel @online_attention_apply_v0_composite
// CHECK: cf.br
// CHECK: ^{{.*}}(%{{.*}}: i32, %{{.*}}: vector<16xf32>, %{{.*}}: vector<16xf32>, %{{.*}}: vector<16xf32>)
// CHECK: cf.cond_br
// CHECK: vc4kernel.tmu_load_fragment
// CHECK: vc4kernel.fragment_reduce
// CHECK-SAME: kind = #vc4kernel.reduce<fmax>
// CHECK: vc4kernel.fragment_cmp
// CHECK-SAME: predicate = #vc4kernel.cmp<ogt>
// CHECK: vc4kernel.fragment_sfu
// CHECK-SAME: kind = #vc4kernel.sfu_kind<exp>
// CHECK: vc4kernel.fragment_reduce
// CHECK-SAME: kind = #vc4kernel.reduce<add>
// CHECK: vc4kernel.tmu_load_fragment
// CHECK: vc4kernel.fragment_reduce
// CHECK-SAME: kind = #vc4kernel.reduce<add>
// CHECK: vc4kernel.fragment_sfu
// CHECK-SAME: kind = #vc4kernel.sfu_kind<recip>
// CHECK: vc4kernel.vdw_store_fragment
// CHECK-NOT: math.
// CHECK-NOT: vector.
// CHECK-NOT: memref.
// CHECK-NOT: func.func
// CHECK-NOT: vc4value.
