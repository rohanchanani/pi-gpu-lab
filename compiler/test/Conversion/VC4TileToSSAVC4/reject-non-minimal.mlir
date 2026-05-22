// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck %s

vc4tile.kernel @not_yet attributes {
  public_name = "not_yet"
} {
  // CHECK: cannot be lowered by the M4 lowering skeleton
  %pid = vc4tile.program_id : i32
  vc4tile.return
}
