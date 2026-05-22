// RUN: vc4-opt --convert-vc4tile-to-ssavc4 %s | FileCheck %s

// CHECK-LABEL: ssavc4.module @vc4tile_lowered
// CHECK-LABEL: ssavc4.func @minimal
// CHECK-SAME: threading = #vc4.threading_mode<single>
// CHECK-SAME: vc4.launch_abi =
// CHECK-SAME: builtins = [{{.*}}name = "num_qpus"{{.*}}uniform_index = 0 : i32{{.*}}]
// CHECK-SAME: tail_policy = "exact_multiple"
// CHECK-SAME: uniform_words_per_qpu = 1 : i32
// CHECK-SAME: vc4.resource
// CHECK: ssavc4.thread_end
vc4tile.kernel @minimal attributes {
  public_name = "minimal_vc4tile"
} {
  vc4tile.return
}
