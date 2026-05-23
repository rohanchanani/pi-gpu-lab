// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck --check-prefix=SSAVC4 %s
// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | vc4-opt --convert-ssavc4-to-vc4 | FileCheck --check-prefix=VC4 %s

// SSAVC4-LABEL: ssavc4.func @minimal_thrend_vc4tile
// SSAVC4-SAME: vc4.launch_abi =
// SSAVC4-SAME: builtins = []
// SSAVC4-SAME: tail_policy = "exact_multiple"
// SSAVC4-SAME: uniform_words_per_qpu = 0 : i32
// SSAVC4-SAME: vc4.resource
// SSAVC4-NOT: ssavc4.uniform.read
// SSAVC4: ssavc4.thread_end

// VC4-LABEL: vc4.func @minimal_thrend_vc4tile
// VC4-SAME: vc4.launch_abi =
// VC4-SAME: builtins = []
// VC4-SAME: tail_policy = "exact_multiple"
// VC4-SAME: uniform_words_per_qpu = 0 : i32
// VC4-SAME: vc4.resource
// VC4: vc4.qpu.bundle
vc4tile.kernel @minimal_thrend_vc4tile attributes {
  public_name = "minimal_thrend_vc4tile"
} {
  vc4tile.return
}
