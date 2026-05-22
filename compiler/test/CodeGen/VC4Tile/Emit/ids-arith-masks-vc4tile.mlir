// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck --check-prefix=SSAVC4 %s
// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 | FileCheck --check-prefix=VC4 %s

// SSAVC4-LABEL: ssavc4.func @ids_arith_masks_vc4tile
// SSAVC4: ssavc4.uniform.read
// SSAVC4: ssavc4.element_number
// SSAVC4: ssavc4.splat
// SSAVC4: ssavc4.alu.add
// SSAVC4: ssavc4.thread_end
// VC4-LABEL: vc4.func @ids_arith_masks_vc4tile
// VC4: vc4.qpu.bundle
vc4tile.kernel @ids_arith_masks_vc4tile attributes {public_name = "ids_arith_masks_vc4tile"} {
  %pid = vc4tile.program_id : i32
  %lanes = vc4tile.lane_range : vector<16xi32>
  %pid_vec = vector.broadcast %pid : i32 to vector<16xi32>
  %sum = arith.addi %lanes, %pid_vec : vector<16xi32>
  vc4tile.return
}
