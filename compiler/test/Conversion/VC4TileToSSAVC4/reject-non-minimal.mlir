// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck %s

vc4tile.kernel @not_yet attributes {
  public_name = "not_yet"
} {
  %base = vc4tile.program_id : i32
  %offsets = vc4tile.lane_range : vector<16xi32>
  %mask = vc4tile.mask_all : vector<16xi1>
  // CHECK: is not supported yet by --convert-vc4tile-to-ssavc4 in this M4 slice
  %value = vc4tile.masked_load_global %base, %offsets, %mask {
    elem_bytes = 4 : i32,
    offset_unit = #vc4tile.offset_unit<byte>
  } : i32, vector<16xi32>, vector<16xi1> -> vector<16xi32>
  vc4tile.return
}
