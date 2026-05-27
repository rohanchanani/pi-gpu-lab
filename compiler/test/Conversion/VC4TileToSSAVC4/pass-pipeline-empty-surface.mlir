// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 | FileCheck %s

// CHECK-LABEL: ssavc4.module @vc4tile_lowered
// CHECK-LABEL: ssavc4.func @pass_pipeline_empty_surface
// CHECK-SAME: vc4.launch_abi =
// CHECK-SAME: builtins = []
// CHECK-SAME: uniform_words_per_qpu = 0 : i32
// CHECK-NOT: vc4tile.surface_placeholder
// CHECK: ssavc4.thread_end
vc4tile.kernel @pass_pipeline_empty_surface attributes {
  public_name = "pass_pipeline_empty_surface"
} {
  vc4tile.surface_placeholder
  vc4tile.return
}
