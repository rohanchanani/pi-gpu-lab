// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_independent attributes {
    public_name = "bad_independent",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 2 : i32
  } {
    // CHECK: independent_vector kernels must have warps_per_block = 1
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_cooperative_low attributes {
    public_name = "bad_cooperative_low",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [],
    warps_per_block = 0 : i32
  } {
    // CHECK: cooperative_block kernels require warps_per_block in range [1, 12]
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_cooperative_high attributes {
    public_name = "bad_cooperative_high",
    schedule_mode = #vc4kernel.schedule_mode<cooperative_block>,
    arg_attrs = [],
    warps_per_block = 13 : i32
  } {
    // CHECK: cooperative_block kernels require warps_per_block in range [1, 12]
    vc4kernel.return
  }
}
