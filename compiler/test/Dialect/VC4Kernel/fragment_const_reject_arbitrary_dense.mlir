// RUN: not vc4-opt %s --verify-vc4kernel 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @fragment_const_reject_arbitrary_dense() attributes {
    public_name = "fragment_const_reject_arbitrary_dense",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    // CHECK: requires efficient fragment_const materialization
    %bad = vc4kernel.fragment_const {value = dense<[4, 0, 7, 1, 10, 2, 13, 3, 16, 4, 19, 5, 22, 6, 25, 7]> : vector<16xi32>} : vector<16xi32>
    vc4kernel.return
  }
}
