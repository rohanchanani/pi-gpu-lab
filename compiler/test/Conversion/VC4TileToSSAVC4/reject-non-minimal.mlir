// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck %s

vc4tile.kernel @not_yet attributes {
  public_name = "not_yet",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>
} {
  // CHECK: is not supported yet by --convert-vc4tile-to-ssavc4 in this M4 slice
  %block = vc4tile.block_id : i32
  vc4tile.return
}
