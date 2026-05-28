// RUN: vc4-opt %S/../../CodeGen/VC4Tile/Hardware/Run/cute_gemm_16x16x16_shared_b_rhs_vc4tile/input.mlir --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck %s

// CHECK-LABEL: ssavc4.func @cute_gemm_16x16x16_shared_b_rhs_vc4tile
// CHECK-SAME: #vc4.builtin_kind<logical_request>
// CHECK-NOT: scf.
// CHECK-NOT: vc4tile.tile_matmul
// CHECK-NOT: vc4tile.tile_contract
// CHECK-NOT: vc4tile.tile_load
// CHECK-NOT: vc4tile.tile_store
// CHECK-NOT: vc4tile.copy_tile
// CHECK: %[[VPM_BASE:.*]] = ssavc4.uniform.read 7 : i32
// CHECK: ssavc4.tmu.request
// CHECK: ssavc4.tmu.read
// CHECK: ssavc4.vdr.load %{{.*}}, %[[VPM_BASE]]
// CHECK: ssavc4.barrier
// CHECK: ssavc4.vpm.read
// CHECK: ssavc4.alu.mul
// CHECK: #vc4.add_opcode<asr>
// CHECK: #vc4.add_opcode<shl>
// CHECK: #vc4.add_opcode<and>
// CHECK: #vc4.add_opcode<sub>
// CHECK: ssavc4.rotate
// CHECK: ssavc4.vdw.store
