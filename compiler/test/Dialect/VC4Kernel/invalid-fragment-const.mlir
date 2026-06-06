// RUN: not vc4-opt %s --verify-vc4kernel --split-input-file 2>&1 | FileCheck %s

module {
  vc4kernel.kernel @bad_arbitrary_dense() attributes {
    public_name = "bad_arbitrary_dense",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    // CHECK: requires efficient fragment_const materialization
    %bad = vc4kernel.fragment_const {value = dense<[0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15]> : vector<16xi32>} : vector<16xi32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_dense_f32() attributes {
    public_name = "bad_dense_f32",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    // CHECK: requires efficient fragment_const materialization
    %bad = vc4kernel.fragment_const {value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01]> : vector<16xf32>} : vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_inf_f32() attributes {
    public_name = "bad_inf_f32",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    // CHECK: fragment_const f32 splat must be finite in P2
    %bad = vc4kernel.fragment_const {value = dense<0x7F800000> : vector<16xf32>} : vector<16xf32>
    vc4kernel.return
  }
}

// -----

module {
  vc4kernel.kernel @bad_shape_mismatch() attributes {
    public_name = "bad_shape_mismatch",
    schedule_mode = #vc4kernel.schedule_mode<independent_vector>,
    arg_attrs = [],
    warps_per_block = 1 : i32
  } {
    // CHECK: value attr type must exactly match result type
    %bad = vc4kernel.fragment_const {value = dense<0> : vector<8xi32>} : vector<16xi32>
    vc4kernel.return
  }
}
