// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck %s

vc4tile.kernel @bad_reduce_kind attributes {public_name = "bad_reduce_kind"} {
  %values = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  // CHECK: currently lowers only add reductions through SSAVC4 rotate/add
  %sum = vc4tile.reduce %values, %mask {kind = #vc4tile.reduce_kind<bit_xor>} : vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.return
}
