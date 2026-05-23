// RUN: not vc4-opt --convert-vc4tile-to-ssavc4 %s 2>&1 | FileCheck %s

vc4tile.kernel @not_yet attributes {
  public_name = "not_yet",
  schedule_mode = #vc4tile.schedule_mode<cooperative_block>,
  uses_barrier = true,
  require_full_block_residency = true,
  semaphores_per_block = 4 : i32
} {
  // CHECK: is not supported yet by --convert-vc4tile-to-ssavc4 in this M4 slice
  vc4tile.barrier {scope = #vc4tile.barrier_scope<block>}
  vc4tile.return
}
