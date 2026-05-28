// RUN: vc4-opt %S/../Hardware/Run/cute_gemm_8x8x8_tiled_vc4tile/input.mlir --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 -o - | FileCheck --check-prefix=SSAVC4 %s
// RUN: vc4-opt %S/../Hardware/Run/cute_gemm_8x8x8_tiled_vc4tile/input.mlir --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 | vc4-opt --convert-ssavc4-to-vc4 --vc4-verify-emit-contract --vc4-verify-scheduled-hardware-rules --vc4-verify-scheduled-adjacent-hazards --vc4-verify-scheduled-io-spacing --vc4-verify-scheduled-peripheral-accesses | FileCheck --check-prefix=VC4 %s

// SSAVC4-LABEL: ssavc4.func @cute_gemm_8x8x8_tiled_vc4tile
// SSAVC4-NOT: scf.
// SSAVC4-NOT: vc4tile.tile_matmul
// SSAVC4-NOT: vc4tile.tile_load
// SSAVC4-NOT: vc4tile.tile_store
// SSAVC4-NOT: ssavc4.cond_br
// SSAVC4: ssavc4.tmu.request
// SSAVC4: ssavc4.alu.mul
// SSAVC4: #vc4.add_opcode<asr>
// SSAVC4: #vc4.add_opcode<shl>
// SSAVC4: #vc4.add_opcode<and>
// SSAVC4: #vc4.add_opcode<sub>
// SSAVC4: ssavc4.rotate
// SSAVC4: ssavc4.vdw.store
// VC4-LABEL: vc4.func @cute_gemm_8x8x8_tiled_vc4tile
// VC4-NOT: ssavc4.
// VC4-NOT: vc4tile.
// VC4: vc4.qpu.bundle
// VC4: #vc4.qpu_signal<thrend>
