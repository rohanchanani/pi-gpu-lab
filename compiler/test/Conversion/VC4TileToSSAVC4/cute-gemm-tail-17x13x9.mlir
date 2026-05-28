// RUN: vc4-opt %S/../../CodeGen/VC4Tile/Hardware/Run/cute_gemm_tail_17x13x9_vc4tile/input.mlir --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s

// CHECK-LABEL: ssavc4.func @cute_gemm_tail_17x13x9_vc4tile
// CHECK-SAME: #vc4.builtin_kind<logical_request>
// CHECK-NOT: scf.
// CHECK-NOT: vc4tile.tile_matmul
// CHECK-NOT: vc4tile.tile_contract
// CHECK-NOT: vc4tile.tile_load
// CHECK-NOT: vc4tile.tile_store
// CHECK-NOT: vc4tile.tile_rect_mask
// CHECK-NOT: vc4tile.masked_load_global
// CHECK: values = array<i32: 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0>
// CHECK: values = array<i32: 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>
// CHECK: ssavc4.tmu.request
// CHECK: ssavc4.tmu.read
// CHECK: ssavc4.make_flags
// CHECK-SAME: kind = #ssavc4.flag_kind<zero_test>
// CHECK: ssavc4.cond_select
// CHECK-SAME: cond = #vc4.cond<zc>
// CHECK: ssavc4.cond_br
// CHECK: ssavc4.vdw.store
