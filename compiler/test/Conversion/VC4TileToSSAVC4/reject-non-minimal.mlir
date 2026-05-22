// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck %s

vc4tile.kernel @not_yet attributes {
  public_name = "not_yet"
} {
  %values = vc4tile.lane_range : vector<16xi32>
  // CHECK: is not supported yet by --convert-vc4tile-to-ssavc4 in this M4 slice
  %rot = vc4tile.rotate %values {amount = 3 : i32} : vector<16xi32> -> vector<16xi32>
  vc4tile.return
}
