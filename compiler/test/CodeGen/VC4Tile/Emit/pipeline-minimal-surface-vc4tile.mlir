// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 | FileCheck --check-prefix=SSAVC4 %s
// RUN: vc4-opt %s --canonicalize-vc4tile-surface --plan-vc4tile-copies --legalize-vc4tile-core-cfg --verify-vc4tile-core --convert-vc4tile-to-ssavc4 | vc4-opt --convert-ssavc4-to-vc4 | FileCheck --check-prefix=VC4 %s

// SSAVC4-LABEL: ssavc4.func @pipeline_minimal_surface_vc4tile
// SSAVC4-SAME: vc4.launch_abi =
// SSAVC4-SAME: builtins = []
// SSAVC4-SAME: uniform_words_per_qpu = 0 : i32
// SSAVC4-NOT: vc4tile.surface_placeholder
// SSAVC4: ssavc4.thread_end

// VC4-LABEL: vc4.func @pipeline_minimal_surface_vc4tile
// VC4-SAME: vc4.launch_abi =
// VC4-SAME: builtins = []
// VC4-SAME: uniform_words_per_qpu = 0 : i32
// VC4: vc4.qpu.bundle
vc4tile.kernel @pipeline_minimal_surface_vc4tile attributes {
  public_name = "pipeline_minimal_surface_vc4tile"
} {
  vc4tile.surface_placeholder
  vc4tile.return
}
