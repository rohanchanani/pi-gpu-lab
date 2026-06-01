// RUN: vc4-opt %s --verify-vc4kernel --convert-vc4kernel-to-ssavc4 | FileCheck %s
// CHECK-LABEL: ssavc4.module @vc4kernel_lowered
// CHECK-LABEL: ssavc4.func @minimal
// CHECK-SAME: vc4.launch_abi
// CHECK-SAME: builtins = []
// CHECK-SAME: uniform_words_per_qpu = 0 : i32
// CHECK-NOT: vc4kernel.
// CHECK: ssavc4.thread_end
module {
  vc4kernel.kernel @minimal attributes {
    public_name = "minimal",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    vc4kernel.return
  }
}
