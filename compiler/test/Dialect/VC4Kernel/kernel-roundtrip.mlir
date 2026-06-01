// RUN: vc4-opt %s --verify-vc4kernel | FileCheck %s
// RUN: vc4-opt %s --verify-vc4kernel --mlir-print-op-generic | FileCheck %s --check-prefix=GENERIC
// CHECK-LABEL: vc4kernel.kernel @minimal
// CHECK-NOT: <invalid-symbol>
// CHECK: public_name = "minimal"
// CHECK: vc4kernel.return
// GENERIC: "vc4kernel.kernel"
// GENERIC: function_type = () -> ()
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
